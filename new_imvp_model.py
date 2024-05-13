import torch
from torch import nn

class SeparablePINN(nn.Module):
    def __init__(self, H=32, W=64):
        super().__init__()
        self.W = W
        self.H = H
        self.pde_points_x = torch.linspace(0, self.W, 64).requires_grad_()
        self.pde_points_y = torch.linspace(0, self.H, 32).requires_grad_()
        self.grid_y, self.grid_x = torch.meshgrid(self.pde_points_y, self.pde_points_x, indexing='ij')

        # self.u_network = nn.Linear(2048, 2048)
        self.u_network = nn.Sequential(
                                        nn.Linear(64, 32),
                                        nn.GELU(),
                                        nn.Linear(32, 64),
                                    )
        self.v_network = nn.Sequential(
                                        nn.Linear(64, 32),
                                        nn.GELU(),
                                        nn.Linear(32, 64),
                                    )
        # self.v_network = nn.Linear(2048, 2048)

    def forward(self, data, calculate_pde=True):
        batch_size, C, H, W = data.shape
        grid_x = self.grid_x.repeat(batch_size, 1, 1).requires_grad_().unsqueeze(1).to(data.device)
        grid_y = self.grid_y.repeat(batch_size, 1, 1).requires_grad_().unsqueeze(1).to(data.device)
        U = data[:, 0].requires_grad_().unsqueeze(1)
        V = data[:, 1].requires_grad_().unsqueeze(1)
        V = V * grid_y
        U = U * grid_x

        delta_U = self.u_network(U)
        delta_V = self.v_network(V)

        if calculate_pde:
            print('calculate_pde')
            du_dx = torch.autograd.grad((U + delta_U).sum(), grid_x, create_graph=True, only_inputs=True,
                                        allow_unused=False)[0]

            dv_dy = torch.autograd.grad((V + delta_V).sum(), grid_y, create_graph=True,
                                       only_inputs=True,allow_unused=False)[0]

            pde_loss = du_dx + dv_dy

            return delta_U, delta_V, torch.nn.functional.l1_loss(pde_loss, torch.zeros_like(pde_loss))
        else:
            # return U.reshape(batch_size, H, W).unsqueeze(1), V.reshape(batch_size, H, W).unsqueeze(1)
            return delta_U, delta_V