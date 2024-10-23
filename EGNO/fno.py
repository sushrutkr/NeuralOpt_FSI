import hydra
from omegaconf import DictConfig
from math import ceil
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader

from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow

from sklearn.preprocessing import MinMaxScaler

torch.cuda.empty_cache()

class FNOData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def wake_trainer(cfg: DictConfig) -> None:
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # Initialize monitoring
    log = PythonLogger(name="wake_fno")
    log.file_logging()
    initialize_mlflow(
        experiment_name=f"wake_FNO",
        experiment_desc=f"training an FNO model for the Flapping Wak",
        run_name=f"wake FNO training",
        run_desc=f"training FNO for Flapping Wake",
        user_name="Sushrut",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # Define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)

    print("Model initialized:", model)

    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )
    
    # Load Data
    scalar = MinMaxScaler(feature_range=(0,1))
    data = np.load("data.npy")
    shape = data.shape
    data = data.reshape(shape[0], -1)
    data = scalar.fit_transform(data)
    data = data.reshape(shape)

    x = data[:,1:,:,:,:]
    y = data[:,:1,:,:,:]

    # Reshape data to 2D array for scaling
    # shape = data.shape
    # y_shape = y.shape
    # data = data.reshape(shape[0], -1)
    # y = y.reshape(y_shape[0], -1)

    # # Reshape back to original shape
    # x = x.reshape(x_shape)
    # y = y.reshape(y_shape)

    # Convert to torch tensors
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    print("x.shape:", x.shape)
    print("y.shape:", y.shape)

    train_dataset = FNOData(x, y)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # Calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)

    print("steps_per_pseudo_epoch:", steps_per_pseudo_epoch)
    print("validation_iters:", validation_iters)

    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of batch size: {validation_iters*cfg.training.batch_size}"
        )

    # Define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")

    for pseudo_epoch in range(max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for batch_index, batch in zip(range(steps_per_pseudo_epoch), train_loader):
                x_batch, y_batch = batch
                loss = forward_train(x_batch.to(dist.device), y_batch.to(dist.device))
                logger.log_minibatch({"loss": loss.detach()})
                print(f"Batch {batch_index}: loss = {loss.item()}")
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # Validation step (if needed)
        # if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
        #     with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
        #         total_loss = 0.0
        #         for _, batch in zip(range(validation_iters), dataloader):
        #             val_loss = validator.compare(
        #                 batch["permeability"],
        #                 batch["wake"],
        #                 forward_eval(batch["permeability"]),
        #                 pseudo_epoch,
        #                 logger,
        #             )
        #             total_loss += val_loss
        #         logger.log_epoch({"Validation error": total_loss / validation_iters})

        # Update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

if __name__ == "__main__":
    wake_trainer()
