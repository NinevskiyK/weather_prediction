import torch
import math
from torch import nn
import torch.nn.functional as F
from pde.models.utils_for_models.imvp_modules import ConvSC, ConvNeXt_block, Learnable_Filter, Attention, ConvNeXt_bottle
from pde.models.utils_for_models.imvp_modules import LayerNorm
from .new_imvp_model import SeparablePINN
from timm.models.layers import trunc_normal_
from einops import rearrange
from pde.utils.plots_graphs import plot_results

import lightning as L

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1).to(device)
        return emb


class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.learnable_emb = LearnablePosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x) + self.learnable_emb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class LearnablePosEmb(nn.Module):
    def __init__(self, dim, max_time_steps=72):
        super().__init__()
        self.dim = dim
        self.time_emb = nn.Parameter(torch.randn(max_time_steps, dim))

    def forward(self, x):
        time_indices = torch.clamp(x.long(), 0, len(self.time_emb) - 1)
        return self.time_emb[time_indices]



def stride_generator(N, T=12, reverse=False):
    strides = [1, 2] * T
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class LP(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(LP, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        super(Predictor, self).__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for i in range(0, N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x, time_emb):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, self.N_T):
            z = self.st_block[i](z, time_emb)

        y = z.reshape(B, int(T / 2), C, H, W)
        return y

class ImprovedLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.norm0 = nn.GroupNorm(num_groups=dim, num_channels=dim)  # Нормализация после первой свертки
        # Дополнительные улучшения...
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.norm1 = nn.GroupNorm(num_groups=dim, num_channels=dim)  # Нормализация после пространственной свертки
        # Дополнительные улучшения...
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.se_block = SEBlock(dim)  # Squeeze-and-Excitation блок

        # Динамическое взвешивание
        self.weight_conv0 = nn.Parameter(torch.ones(1))
        self.weight_conv1 = nn.Parameter(torch.ones(1))


    def forward(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.norm0(attn)
        attn = self.conv_spatial(attn)
        attn = self.norm1(attn)
        attn = self.conv1(attn)

        # Применение SE блока перед умножением
        attn = self.se_block(attn)

        # Динамическое взвешивание выходов
        attn = self.weight_conv0 * attn + self.weight_conv1 * u

        return u * attn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.projections = nn.ModuleList([ImprovedLKA(self.head_dim) for _ in range(num_heads)])
        self.output_projection = nn.Conv2d(d_model, d_model, 1)
        self.se_block = SEBlock(d_model)  # Включение Squeeze-and-Excitation блока

    def forward(self, x):
        batch_size, _, H, W = x.size()
        x = x.view(batch_size, self.num_heads, self.head_dim, H, W)

        attn_outputs = [proj(x[:, i, :, :, :]) for i, proj in enumerate(self.projections)]
        attn = torch.cat(attn_outputs, dim=1)
        attn = attn.view(batch_size, self.d_model, H, W)

        attn = self.se_block(attn)  # Применение SE блока к объединенным признакам

        return self.output_projection(attn)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, transpose=False,
                 act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if transpose is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        # self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class ConvNeXt_bottle(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(64, dim)
        )
        self.dwconv = nn.Conv2d(dim * 2, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv = LKA(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, time_emb=None):
        input = x
        time_emb = self.mlp(time_emb)
        x = self.dwconv(x) + rearrange(time_emb, 'b c -> b c 1 1')
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = self.res_conv(input) + self.drop_path(x)
        return x


class ConvNeXt_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(64, dim)
        )
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = ImprovedLKA(dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, time_emb=None):
        input = x
        time_emb = self.mlp(time_emb)
        x = self.dwconv(x) + rearrange(time_emb, 'b c -> b c 1 1')
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, T, dim):
        """
        Инициализирует слой для добавления временных позиционных эмбеддингов.

        Аргументы:
        - T (int): Максимальное количество временных шагов.
        - dim (int): Размерность эмбеддинга.
        """
        super(TemporalPositionalEncoding, self).__init__()
        assert dim % 2 == 0, "Размерность должна быть четной"
        self.T = T
        self.dim = dim
        pe = torch.zeros(T, dim)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, B, T, C, H, W):
        """
        Добавляет временные позиционные эмбеддинги к входному тензору.

        Аргументы:
        - x (torch.Tensor): Входной тензор размерности (B*T, C, H, W).

        Возвращает:
        - torch.Tensor: Тензор с добавленными временными позиционными эмбеддингами, размерность не меняется.
        """

        pe = self.pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pe = pe.expand(B, self.T, self.dim, H, W)
        pe = pe.contiguous().view(B * self.T, self.dim, H, W)
        # Обрезаем или расширяем pe, чтобы соответствовать размерности канала входа
        if C != self.dim:
            if C < self.dim:
                pe = pe[:, :C, :, :]
            else:
                pe_temp = torch.zeros(B * self.T, C, H, W, device=x.device)
                pe_temp[:, :self.dim, :, :] = pe
                pe = pe_temp
        return torch.cat((x, pe), dim=1)


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, T=12):
        super(Decoder, self).__init__()
        self.T = T
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        # self.readout = nn.Conv2d(640, 64, 1)
        self.readout = nn.Conv2d(768, 64, 1)

    def forward(self, hid, enc1, latent_1, latent_2, latent_3):
        hid = self.dec[0](hid + latent_3)
        hid = self.dec[1](hid + latent_2)
        hid = self.dec[2](hid + latent_1)

        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        Y = Y.reshape(int(ys[0]/self.T), int(ys[1]*self.T), 32, 64)
        Y = self.readout(Y)
        return Y


class DynamicWeighting(nn.Module):
    def __init__(self):
        super(DynamicWeighting, self).__init__()
        # Инициализация не требует указания входных и выходных каналов, так как мы будем усреднять по каналам и пространственным измерениям

    def forward(self, y_predicted):
        # Предполагаем, что y_predicted имеет размерность [bs, t, c, h, w]

        # Усреднение по каналам и пространственным размерностям для получения значений [bs, t]
        # это сводит всю информацию по каналам и по пространству в единственное скалярное значение
        # для каждого временного шага в каждом примере батча.
        bs, t, c, h, w = y_predicted.shape
        # Сначала выполняем усреднение по каналам, получаем shape = [bs, t, h, w]
        averaged_over_channels = y_predicted.mean(dim=2)
        # Затем усредняем совмещенно по пространственным измерениям, получаем shape = [bs, t]
        averaged_over_hw = averaged_over_channels.mean(dim=[2, 3])

        # Предполагается, что веса в этом случае определяются напрямую из усредненных значений,
        # или можно применять дополнительную операцию, если требуется изменить диапазон или распределение значений
        weights = F.sigmoid(averaged_over_hw)  # Сигмоид для преобразования весов в диапазон [0, 1]

        return weights



class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, T=12):
        super(Decoder, self).__init__()
        self.T = T
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        # self.readout = nn.Conv2d(640, 64, 1)
        self.readout = nn.Conv2d(768, 64, 1)

    def forward(self, hid, enc1, latent_1, latent_2, latent_3):
        hid = self.dec[0](hid + latent_3)
        hid = self.dec[1](hid + latent_2)
        hid = self.dec[2](hid + latent_1)

        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        Y = Y.reshape(int(ys[0]/self.T), int(ys[1]*self.T), 32, 64)
        Y = self.readout(Y)
        return Y

class IAM4VP(L.LightningModule):
    def __init__(self, shape_in, dataset_std, dataset_mean, hid_S=64, hid_T=512, N_S=4, N_T=6, time_prediction=12):
        super().__init__()
        T, C, H, W = shape_in
        self.T = T
        self.time_mlp = Time_MLP(dim=64)
        self.tpe = TemporalPositionalEncoding(self.T, hid_S)

        self.enc = Encoder(2*C, hid_S, N_S)
        self.hid = Predictor(T * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.attn = MultiHeadAttention(64)
        self.readout_u = nn.Conv2d(64, 1, 1)
        self.readout_v = nn.Conv2d(64, 1, 1)
        self.mask_token = nn.Parameter(torch.zeros(T, hid_S, 8, 16))
        self.lp = LP(C, hid_S, N_S)
        self.dynamic_weighting = DynamicWeighting()
        self.SeparablePINN = SeparablePINN(H, W)

        self.time_prediction = time_prediction
        self.criterion = torch.nn.functional.l1_loss
        self.automatic_optimization = False

        self.dataset_std = dataset_std
        self.dataset_mean = dataset_mean

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler_config": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        opt = self.optimizers()
        opt.zero_grad()

        pred_list = []
        for idx_time in range(self.time_prediction):
            t = torch.tensor(idx_time * 100).repeat(x.shape[0]).to(self.device)
            prediction, sep_loss = self.forward(x, pred_list, t)
            pred_list.append(prediction.detach())

            loss = self.criterion(prediction, y[:, idx_time]) + sep_loss
            self.manual_backward(loss)
            opt.step()
            self.log_dict({"train/loss": loss, "train/idx_time": idx_time})
            self.logger.log_metrics({"train/loss": loss, "train/idx_time": idx_time})


    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred_list = []
        for idx_time in range(self.time_prediction):
            t = torch.tensor(idx_time * 100).repeat(x.shape[0]).to(self.device)
            prediction = self.forward(x, pred_list, t, is_train=False)
            pred_list.append(prediction.detach())
            loss = self.criterion(prediction, y[:, idx_time])

            self.log_dict({"valid/loss": loss, "valid/idx_time": idx_time})
            self.logger.log_metrics({"valid/loss": loss, "valid/idx_time": idx_time})
            if batch_idx == 0:
                self.logger.log_image(key="epoch: {}, forward_time: {}".format(self.current_epoch, idx_time),
                                    images=[
                                        plot_results(prediction, y[:, -1], self.dataset_std[0], self.dataset_mean[0], idx_hour=idx_time, name_of_model="imvp")
                                        ])

    def forward(self, x_raw, y_raw=None, t=None, is_train=True):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        x = self.tpe(x, B, T, C, H, W)
        time_emb = self.time_mlp(t)

        embed, skip, embed_1, embed_2 = self.enc(x)

        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)

        if len(y_raw) != 0:
            y_predicted = torch.empty([B, len(y_raw), C, H, W]).to(self.device)
            for i in range(len(y_raw)):
                y_predicted[:, i] = y_raw[i]
            weights = self.dynamic_weighting(y_predicted)

        for idx, pred in enumerate(y_raw):
            embed2, _ = self.lp(pred)
            mask_token[:, idx, :, :, :] = embed2 * weights[:, idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        hid = self.hid(z, time_emb)

        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip, embed_1, embed_2, embed)
        Y = self.attn(Y)
        U = self.readout_u(Y)
        V = self.readout_v(Y)

        if is_train:
            delta_U, delta_V, loss = self.SeparablePINN(torch.stack((U.squeeze(), V.squeeze()), dim=1), calculate_pde=is_train)
        else:
            delta_U, delta_V = self.SeparablePINN(torch.stack((U.squeeze(), V.squeeze()), dim=1), calculate_pde=is_train)

        result = torch.cat((U + delta_U, V + delta_V), dim=1)

        if is_train:
            return result, loss
        else:
            return result