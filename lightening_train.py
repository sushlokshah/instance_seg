from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import Logger, WandbLogger
import os
from typing import Any, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks import Callback
import numpy as np
import cv2

############################################################
# local imports
from dataset.instance_dataloader import Cityscapes
from models.Enet import ENet
from models.resnet_based_model import Custom_model
from loss_functions.cluster_loss import Cluster_loss, Classification_loss

############################################################


############################################################
# defining data module
############################################################
class CityScapes_lighting(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        self.data_dir = data_dir
        # data transformations
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Cityscapes(
                root=self.data_dir,
                split="train",
                mode="fine",
                target_type=["instance"],
                transforms=self.transforms,
                random_crop=True,
                random_flip=0.5,
                random_rotate=10,
            )
            self.data_val = Cityscapes(
                root=self.data_dir,
                split="val",
                mode="fine",
                target_type=["instance"],
                transforms=self.transforms,
                random_crop=True,
            )
            self.data_test = Cityscapes(
                root=self.data_dir,
                split="test",
                mode="fine",
                target_type=["instance"],
                transforms=self.transforms,
                random_crop=True,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


############################################################
# defining model
############################################################
class Instance_Lighting(LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = Custom_model()
        # ENet(out_channels=64)
        self.cluster_loss = Cluster_loss(
            delta_cluster_distance=0.2,
            delta_variance_loss=0.2,
            alpha=1,
            beta=1,
            gamma=0.1,
        )
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def model_step(self, batch: Any):
        x, (inst) = batch
        features = self.forward(x)
        # print(batch_idx)
        # print(features.shape)
        # print(inst.shape)
        loss, cache = self.cluster_loss(features, inst)
        return features, loss, cache

    def training_step(self, batch: Any, batch_idx: int):
        torch.cuda.empty_cache()
        features, loss, cache = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        torch.cuda.empty_cache()
        features, loss, cache = self.model_step(batch)
        cluster_heads = cache[-1]
        masks = self.generate_mask(cluster_heads, features)
        # save fisrt image of batch with masks
        image = batch[0]
        image = image[0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        mask = masks[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        # print(mask.shape)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        cv2.imwrite(
            "/home/awi-docker/video_summarization/instance_seg/dataset/vis/val_mask{}.png".format(
                batch_idx
            ),
            mask,
        )
        cv2.imwrite(
            "/home/awi-docker/video_summarization/instance_seg/dataset/vis/val_image{}.png".format(
                batch_idx
            ),
            image,
        )

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def generate_mask(self, cluster_heads, features):
        """
        :param cluster_heads: (N, C, K)
        :param features: (N, C, H, W)
        :return: (N, K, H, W)
        """
        masks = []
        for n in range(len(cluster_heads)):
            mask = torch.zeros(
                (len(cluster_heads[n][0]), features.shape[2], features.shape[3])
            )
            for k in range(len(cluster_heads[n][0])):
                # print(
                #     (
                #         features[n] * cluster_heads[n][:, k].unsqueeze(1).unsqueeze(1)
                #     ).shape
                # )
                mask[k] = torch.norm(
                    features[n] * cluster_heads[n][:, k].unsqueeze(1).unsqueeze(1),
                    dim=0,
                )
                # print(mask[k].shape)
                # normalize mask
                mask[k] = (mask[k] - torch.min(mask[k])) / (
                    torch.max(mask[k]) - torch.min(mask[k])
                )
                mask[k] = mask[k] > 0.5
                mask[k] = mask[k] * (k + 1)
            final_mask = torch.sum(mask, dim=0)
            final_mask = final_mask / torch.max(final_mask)
            masks.append(final_mask)

        return masks

    def test_step(self, batch: Any, batch_idx: int):
        torch.cuda.empty_cache()
        features, loss, cache = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedular = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, verbose=True
            ),
            "monitor": "val/loss",  # Default: val_loss
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [schedular]


############################################################
# defining callbacks
############################################################
if __name__ == "__main__":
    # set name based on date-time
    import datetime

    now = datetime.datetime.now()
    name = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(name)

    # init data
    data_module = CityScapes_lighting(
        data_dir="/home/awi-docker/video_summarization/instance_seg/dataset/cityscapes",
        batch_size=2,
        num_workers=4,
        pin_memory=True,
    )

    # init model
    model = Instance_Lighting(lr=0.001)

    # init logger
    logger = WandbLogger(
        name=name,
        project="video_summarization",
        log_model=True,
        save_dir="/home/awi-docker/video_summarization/instance_seg/logs/",
    )

    # init trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=100,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="/home/awi-docker/video_summarization/instance_seg/weights/instance_seg-{epoch:02d}-{val_loss:.2f}",
            ),
            # pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            # pl.callbacks.EarlyStopping(
            #     monitor="val/loss",
            #     patience=10,
            #     mode="min",
            #     verbose=True,
            #     check_on_train_epoch_end=True,
            # ),
        ],
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value"
        # resume_from_checkpoint="lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt",
    )

    # train
    trainer.fit(model, data_module)

    # # test
    # trainer.test(model, data_module.test_dataloader())
