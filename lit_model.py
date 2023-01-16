from typing import Any, List

import timm
import torch
import torch.nn.functional as F
from model import BeitV2, Swin, Tiny_Vit_11M, Tiny_Vit_21M, Tiny_Vit_5M, ConvNextV2_femto, EfficientFormerV2
from pytorch_lightning import LightningModule
from sklearn.metrics import classification_report
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss
from torchmetrics import ConfusionMatrix, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from utils import plot_top_losses, save_images_with_confidence, save_json_file
from augment import hflip



def load_model(model_file):
    model = torch.jit.load(model_file)
    model.to('cpu')
    model.eval()
    return model


class LitModule(LightningModule):
    def __init__(
        self,
        net: str,
        lr,
        scheduler=False,
        progressive=False
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        if net == 'swin-small':
            self.net = Swin()
        elif net == 'beitv2':
            self.net = BeitV2()
        elif net == 'tiny_vit_21M':
            self.net = Tiny_Vit_21M(pretrained=True)
        elif net == 'tiny_vit_11M':
            self.net = Tiny_Vit_11M(pretrained=True)
        elif net == 'tiny_vit_5M':
            self.net = Tiny_Vit_5M(pretrained=True)
        elif net.startswith('mobilevitv2'):
            self.net = timm.create_model(net, num_classes=3, pretrained=True)
        elif net == 'convnext':
            self.net = timm.create_model(net, num_classes=3, pretrained=True)
        elif net == "dynamic_quantized":
            self.net = load_model("weights/tinyvit21M_dynamic_quantized_script.pth")
        elif net == "static_quantized_fx":
            self.net = load_model("weights/tinyvit5M_fx_graph_mode_quantized_fbgemm.pth")
        elif net == "script":
            self.net = load_model("weights/tinyvit21M_script.pth")
        elif net == "convnextv2-femto":
            self.net = ConvNextV2_femto()
        elif net == "aware":
            self.net = load_model("weights/tinyvit11M_aware.pth")
        elif net == "efficientformer_l3":
            self.net = timm.create_model(net, num_classes=3, pretrained=True)
            self.net.load_state_dict(torch.load("weights/efficientformer_l3.pth"))
        elif net == "efficientformer_l1":
            self.net = timm.create_model(net, num_classes=3, pretrained=True)
            self.net.load_state_dict(torch.load("weights/efficientformer_l1.pth"))
        elif net == "efficientformer_l1_dynamic":
            self.net = load_model("weights/efficientformer_l1_dynamic_quantized_script.pth")
        elif net == "efficientformer_l3_dynamic":
            self.net = load_model("weights/efficientformer_l3_dynamic_quantized_script.pth")
        elif net == "efficientformerv2_s1":
            self.net = EfficientFormerV2(pretrained=True)

        self.lr = lr
        # loss function
        # self.criterion = LabelSmoothingCrossEntropy(0.1)
        self.criterion = CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.cmt = ConfusionMatrix(3)
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.progressive = progressive
        self.img_confs = {
            "img_name":[],
            "score": [],
            "label": []
        }

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y= batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y


    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)


        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # y_pred = torch.cat([i["preds"] for i in outputs], dim=0)
        # y_true = torch.cat([i["targets"] for i in outputs], dim=0)
        # print("Confusion matrix [train]: ", self.cmt(y_true, y_pred))
        pass
        
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        # y_pred = torch.cat([i["preds"] for i in outputs], dim=0)
        # y_true = torch.cat([i["targets"] for i in outputs], dim=0)
        # print("Confusion matrix [val]: ", self.cmt(y_true, y_pred))

    def debug_step_plot_loss(self, batch:Any):
        x, y = batch
        loss = []
        preds = []
        for idx, _ in enumerate(x):
            logits = self.forward(x[idx].unsqueeze(0))
            loss_sample = self.criterion(logits, y[idx].unsqueeze(0))
            loss.append(loss_sample)
            preds_sample = torch.argmax(logits, dim=1)
            preds.append(preds_sample)
        return loss, preds, y
    
    def debug_step(self, batch: Any):
        x, y, file_names = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        confidence_scores, _ = torch.max(F.softmax(logits, dim=1), dim=1)
        return loss, preds, confidence_scores, y, file_names


    def test_step(self, batch: Any, batch_idx: int):
        # loss, preds, confidence_scores, targets, file_names = self.debug_step(batch)
        # loss, preds, targets = self.debug_step_plot_loss(batch)
        # plot_top_losses(batch[0], batch[1], batch_idx, loss, targets, preds)
        # save_images_with_confidence(self.img_confs, confidence_scores, targets, file_names)
        loss, preds, targets = self.step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        y_pred = torch.cat([i["preds"] for i in outputs], dim=0)
        y_true = torch.cat([i["targets"] for i in outputs], dim=0)
        print("[TEST] Confusion Matrix: ", self.cmt(y_pred, y_true))
        # save_json_file(self.img_confs)
        print("[TEST]Classification report: ", classification_report(y_true.cpu(), y_pred.cpu(), digits=4))
        
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=10e-8, amsgrad=False)
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                            max_lr=0.01, 
                                                            steps_per_epoch=743, 
                                                            epochs=40
                                                            )
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


