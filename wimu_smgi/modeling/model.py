import torch
from torch import nn
import lightning.pytorch as pl
import torchvision.models as models


class MusicGenreClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        conv_out_channels: int = 64,
        conv_kernel_size: int = 7,
        conv_stride: int = 2,
        conv_padding: int = 3
    ):
        super(MusicGenreClassifier, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = models.resnet50(weights=None)

        self.model.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=False
        )
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
