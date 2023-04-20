import torch
import torch_geometric
import pytorch_lightning as pl
import torchmetrics.functional as pl_metrics

from torch.nn.functional import softmax
from typing import Tuple
from .networks import by_name as model_by_name


class RecognitionModel(pl.LightningModule):

    def __init__(self, network, dataset: str, num_classes, img_shape: Tuple[int, int],
                 dim: int = 3, learning_rate: float = 1e-3, weight_decay: float = 5e-3, **model_kwargs):
        super(RecognitionModel, self).__init__()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_outputs = num_classes
        self.dim = dim

        model_input_shape = torch.tensor(img_shape + (dim, ), device=self.device)
        self.model = model_by_name(network)(dataset, model_input_shape, num_outputs=num_classes, **model_kwargs)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # data.pos = data.pos[:, :self.dim]
        # data.edge_attr = data.edge_attr[:, :self.dim]
        return self.model.forward(data)

    ###############################################################################################
    # Steps #######################################################################################
    ###############################################################################################
    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=batch.y)

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = pl_metrics.accuracy(preds=y_prediction, target=batch.y)
        self.logger.log_metrics({"Train/Loss": loss, "Train/Accuracy": accuracy}, step=self.trainer.global_step)
        return loss

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(data=batch)
        y_prediction = torch.argmax(outputs, dim=-1)
        predictions = softmax(outputs, dim=-1)

        self.log("Val/Loss", self.criterion(outputs, target=batch.y))
        self.log("Val/Accuracy", pl_metrics.accuracy(preds=y_prediction, target=batch.y))
        k = min(3, self.num_outputs - 1)
        self.log(f"Val/Accuracy_Top{k}", pl_metrics.accuracy(preds=predictions, target=batch.y, top_k=k))
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRPolicy())
        # return [optimizer], [lr_scheduler]
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr*1.5, epochs=100, steps_per_epoch=205, anneal_strategy='cos', div_factor=2.0, final_div_factor=5.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


class LRPolicy(object):
    def __call__(self, epoch: int):
        if epoch < 20:
            # return 5e-3
            return 1
        elif epoch >= 20 and epoch < 50:
            # return 5e-4
            return 0.5
        else:
            return 0.1
