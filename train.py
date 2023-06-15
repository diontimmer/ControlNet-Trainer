from share import prepare_model_for_training, CustomModelCheckpoint, get_latest_ckpt
import injects  # noqa: F401
from config import config
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.hack import enable_sliced_attention
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import gc


def train_controlnet():
    gc.collect()
    torch.cuda.empty_cache()

    if not os.path.exists(config.logging_dir):
        os.makedirs(config.logging_dir)

    wandb_logger = None
    if config.wandb_key:
        wandb.login(key=config.wandb_key)
        wandb_logger = WandbLogger(
            save_dir=config.logging_dir,
            project=config.project_name,
            name=config.run_name if config.run_name else None,
        )

    if config.save_memory:
        enable_sliced_attention()

    torch.set_float32_matmul_precision("medium")

    run_filename = f"_run_{config.run_name}" if config.run_name else ""

    # ckpt_callback
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=config.output_dir,
        every_n_train_steps=config.save_ckpt_every_n_steps,
        save_weights_only=config.save_weights_only,
        save_top_k=config.save_top_k,
        filename=config.project_name + run_filename + "_{epoch:03d}_{step:06d}",
        save_last=config.save_last,
    )

    # get number of gpus
    num_gpus = torch.cuda.device_count()

    print("Number of GPUs:", num_gpus)
    print("Batch Size:", config.batch_size)
    print("Max Epochs:", config.max_epochs)

    # Data
    dataset = MyDataset()
    print("Dataset size:", len(dataset))
    model = prepare_model_for_training()

    dataloader = DataLoader(
        dataset, num_workers=0, batch_size=config.batch_size, shuffle=True
    )

    logger = ImageLogger(
        batch_frequency=config.image_logger_freq,
        disabled=config.image_logger_disabled,
        wandb_logger=wandb_logger,
    )

    # login to wandb and train!

    trainer = (
        pl.Trainer(
            devices=num_gpus,
            accelerator="gpu",
            precision=32,
            callbacks=[logger, checkpoint_callback],
            log_every_n_steps=1,
            max_epochs=config.max_epochs,
            strategy="ddp_find_unused_parameters_true",
            logger=wandb_logger if wandb_logger else None,
        )
        if config.multi_gpu
        else pl.Trainer(
            devices=1,
            accelerator="gpu",
            precision=32,
            callbacks=[logger, checkpoint_callback],
            log_every_n_steps=1,
            max_epochs=config.max_epochs,
            logger=wandb_logger,
        )
    )

    print("Starting the training process...")

    if config.resume_ckpt == "latest":
        config.resume_ckpt = get_latest_ckpt()

    if config.resume_ckpt:
        if not os.path.exists(config.resume_ckpt):
            print("Checkpoint file does not exist:", config.resume_ckpt)
            config.resume_ckpt = None

    trainer.fit(
        model,
        dataloader,
        ckpt_path=None if not config.resume_ckpt else config.resume_ckpt,
    )

    print("Training completed!")


if __name__ == "__main__":
    train_controlnet()
