import os
import requests
import tarfile
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datasets import DrinksDataset
from config import pretrained_model_dir
from utils import collate_fn, evaluate_iou
from config import default_config
from engine import evaluate


class FasterRCNNModelTest(LightningModule):
    def __init__(self, num_classes=4, lr=0.005, batch_size=4):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        #self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.cuda()

    def forward(self, x):
        self.model.cuda().eval()
        return self.model(x)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        cft = 0.6
        iou = torch.stack([evaluate_iou(t, o, cft) for t, o in zip(targets, preds)]).mean()
        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        apr_dict = metric.compute()
        return {"iou": iou, "pred": preds, "map": apr_dict["map"], "mar_100": apr_dict["mar_100"]}

    def test_epoch_end(self, outputs):
        avg_iou = torch.stack([o["iou"] for o in outputs]).mean()
        mAP = torch.stack([o["map"] for o in outputs]).mean()
        mar_100 = torch.stack([o["mar_100"] for o in outputs]).mean()
        self.log("test_epoch_iou", avg_iou)
        self.log("test_epoch_map", mAP)
        self.log("test_epoch_mar_100", mar_100)
        return {"test_iou": avg_iou, "test_map": mAP, "test_mar_100": mar_100}
    
    def setup(self, stage=None):
        self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(DrinksDataset(default_config["test_split"]),
                         batch_size=default_config['batch_size'],
                         shuffle=False,
                         num_workers=default_config['num_workers'],
                         pin_memory=default_config['pin_memory'],
                         collate_fn=collate_fn)

if __name__ == '__main__':

    if not os.path.exists(default_config["test_split"]):
        fname = "drinks.tar.gz"
        url = f'https://github.com/DJSapit/Drinks-Dataset-Faster-RCNN-Sapit/releases/download/v0.1.0-alpha/{fname}'
        print(f'downloading drinks dataset from {url}')
        r = requests.get(url, allow_redirects=True)
        with open('drinks.tar.gz', 'wb') as file:
            file.write(r.content)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
        if os.path.exists(default_config["test_split"]):
            print("drinks dataset downloaded successfully")
        else:
            raise Exception("drinks dataset download failed")

    if not os.path.exists(pretrained_model_dir):
        os.makedirs(os.path.dirname(pretrained_model_dir), exist_ok=True)
        fname = os.path.basename(pretrained_model_dir)
        url = f'https://github.com/DJSapit/Drinks-Dataset-Faster-RCNN-Sapit/releases/download/v0.1.0-alpha/{fname}'
        print(f'downloading pretrained model from {url}')
        r = requests.get(url, allow_redirects=True)
        with open(pretrained_model_dir, 'wb') as file:
            file.write(r.content)
        if os.path.exists(pretrained_model_dir):
            print("pretrained model downloaded successfully")
        else:
            raise Exception("pretrained model download failed")

    print(f"loading model from {pretrained_model_dir}")

    model = FasterRCNNModelTest.load_from_checkpoint(pretrained_model_dir)
    model.cuda().eval()
    trainer = Trainer(enable_model_summary=False, accelerator="auto", devices="auto", max_epochs=1)
    trainer.test(model)
    test_data = DataLoader(DrinksDataset(default_config["test_split"]),
                         batch_size=default_config['batch_size'],
                         shuffle=False,
                         num_workers=default_config['num_workers'],
                         pin_memory=default_config['pin_memory'],
                         collate_fn=collate_fn)
    evaluate(model, test_data, device=torch.device("cuda"))

