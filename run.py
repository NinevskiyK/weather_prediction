import os

import warnings
warnings.filterwarnings('ignore')

from imvp_lightning import IAM4VP
from pde.train_utils.dataloader_openstl import load_data

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

data_root = "/home/fa.buzaev/data_5/"
prediction_horizone = 12

dataloader_train, dataloader_vali, dataloader_test, mean, std = load_data(batch_size=100,
                                                                        val_batch_size=100,
                                                                        data_root=data_root,
                                                                        num_workers=6,
                                                                        data_split='5_625',
                                                                        data_name=['u10', 'v10'],
                                                                        train_time=['2010', '2015'],
                                                                        val_time=['2016', '2016'],
                                                                        test_time=['2017', '2018'],
                                                                        idx_in=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
                                                                        idx_out=[*range(1, prediction_horizone+1)],
                                                                        step=1,
                                                                        levels=['50'],
                                                                        distributed=False, use_augment=False,
                                                                        use_prefetcher=False, drop_last=False)

model = IAM4VP(shape_in=[12, 2, 32, 64], dataset_std=std, dataset_mean=mean, time_prediction=prediction_horizone)

checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="valid/loss",
    mode="min",
    dirpath="/home/knninevskiy/checkpoints",
    filename="weatherbench-dist-{epoch:02d}-{valid/loss:.2f}",
    every_n_epochs=1
)

os.environ["SLURM_JOB_NAME"]="bash"
wandb_logger = WandbLogger(project="WeatherPrediction", name="multigpu_training_450")

trainer = L.Trainer(max_epochs=50, accelerator='cuda', callbacks=[checkpoint_callback, RichProgressBar()], logger=wandb_logger, enable_progress_bar=True, devices=2, strategy="ddp_find_unused_parameters_true")
trainer.fit(model, dataloader_train, dataloader_vali)
