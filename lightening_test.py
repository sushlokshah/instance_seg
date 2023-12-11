import os
from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as L
from loss_functions.cluster_loss import Cluster_loss, Classification_loss
from models.Enet import ENet

import torch


# define the LightningModule
class Instance_seg(L.LightningModule):
    def __init__(self, model, alpha=1, beta=1, ema_weight=1):
        super().__init__()
        self.model = model
        self.ema_model = model
        self.ema_weight = ema_weight
        self.alpha = alpha
        self.beta = beta
        # self.classification_model = classification_model
        self.cluster_loss = Cluster_loss()
        # self.classification_loss = Classification_loss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        out = self.model(x)
        loss, cache = self.cluster_loss(out, y)
        # classes = self.classification_model(cluster_head)
        # classification_loss = self.classification_loss(classes, y)

        total_loss = self.alpha * loss  # + self.beta * classification_loss

        # Logging to TensorBoard by default
        self.log("train_loss", total_loss)
        self.log("cluster_loss", loss)
        # self.log("classification_loss", classification_loss)

        if self.ema_weight > 0:
            with torch.no_grad():
                for ema_param, param in zip(
                    self.ema_model.parameters(), self.model.parameters()
                ):
                    ema_param.data.mul_(self.ema_weight).add_(
                        (1 - self.ema_weight) * param.data
                    )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss, cache = self.cluster_loss(out, y)
        # classes = self.classification_model(cluster_head)
        # classification_loss = self.classification_loss(classes, y)

        total_loss = self.alpha * loss  # + self.beta * classification_loss

        self.log("val_loss", total_loss)
        self.log("cluster_loss", loss)
        # self.log("classification_loss", classification_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = nn.functional.mse_loss(out, x)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # also add lr scheduler
        schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        return [optimizer], [schedular]

    # log lr on each epoch
    def on_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]
        self.logger.experiment.add_scalar("lr", lr, self.current_epoch)


# define callbacks


def load_callbacks():
    callbacks = []
    callbacks.append(L.callbacks.ModelCheckpoint(monitor="val_loss"))
    callbacks.append(L.callbacks.LearningRateMonitor())
    callbacks.append(L.callbacks.GPUStatsMonitor())
    callbacks.append(L.callbacks.ProgressBar())
    callbacks.append(L.callbacks.EarlyStopping(monitor="val_loss"))
    callbacks.append(
        L.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{val_loss:.2f}",
            save_last=True,
        )
    )
    return callbacks


dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
dataloader = utils.data.DataLoader(dataset, batch_size=32)

model = ENet()
classification_model = ENet()

# init model
model = Instance_seg(model, classification_model)
