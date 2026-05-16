import torch
from torch import nn
import lightning.pytorch as pl
import torchvision.models as models
import torchmetrics


class MusicGenreClassifier(pl.LightningModule):
    def __init__(
        self,
        genre_names: list,
        learning_rate: float = 1e-3,
        conv_out_channels: int = 64,
        conv_kernel_size: int = 7,
        conv_stride: int = 2,
        conv_padding: int = 3
    ):
        super(MusicGenreClassifier, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.genres = genre_names
        self.model = models.resnet18(weights=None)
        num_classes = len(self.genres)

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

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.acc_per_class = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="none")
        self.precision_per_class = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="none")
        self.recall_per_class = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="none")
        self.f1_per_class = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="none")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        loss = self.criterion(outputs, labels)

        y_true_indices = torch.argmax(labels, dim=1)
        acc = self.accuracy(outputs, y_true_indices)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        y_true_indices = torch.argmax(labels, dim=1)
        acc = self.accuracy(outputs, y_true_indices)
        self.log('val_loss', loss)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.acc_per_class(outputs, y_true_indices)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        y_true_indices = torch.argmax(labels, dim=1)
        acc = self.accuracy(outputs, y_true_indices)
        
        self.log('test_loss', loss)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        
        self.acc_per_class(outputs, y_true_indices)
        self.precision_per_class(outputs, y_true_indices)
        self.recall_per_class(outputs, y_true_indices)
        self.f1_per_class(outputs, y_true_indices)

    def on_test_epoch_end(self):
        class_accuracies = self.acc_per_class.compute()
        class_precisions = self.precision_per_class.compute()
        class_recalls = self.recall_per_class.compute()
        class_f1s = self.f1_per_class.compute()
        
        for i, genre_name in enumerate(self.genres):
            self.log(f"test_acc/{genre_name}", class_accuracies[i])
            self.log(f"test_precision/{genre_name}", class_precisions[i])
            self.log(f"test_recall/{genre_name}", class_recalls[i])
            self.log(f"test_f1/{genre_name}", class_f1s[i])
            
        self.acc_per_class.reset()
        self.precision_per_class.reset()
        self.recall_per_class.reset()
        self.f1_per_class.reset()