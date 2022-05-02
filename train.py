import os
import requests
import tarfile
import torch
import multiprocessing as mp
import wandb
from argparse import ArgumentParser
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
import torchvision.transforms as T
from utils import collate_fn, evaluate_iou, pred2boxes, target2boxes
from datasets import DrinksDataset
from config import project_name, default_config
from engine import evaluate


def get_args():
    parser = ArgumentParser(description="PyTorch Lightning Drinks Dataset test")
    parser.add_argument("--precision", type=int,    default=default_config["precision"], choices=[16, 32], help="use 32 or 16 bit precision in training, "
                                                                                                                f"default = {default_config['precision']}")
    parser.add_argument("--max-epochs", type=int,   default=default_config["max_epochs"], help=f"default = {default_config['max_epochs']}")
    parser.add_argument("--batch-size", type=int,   default=default_config["batch_size"], help=f"default = {default_config['batch_size']}")
    parser.add_argument("--lr", type=float,         default=default_config["lr"], help=f"learning rate, default = {default_config['lr']}")
    parser.add_argument("--num-workers", type=int,  default=default_config["num_workers"], help=f"default = {default_config['num_workers']}")
    parser.add_argument("--num-classes", type=int,  default=default_config["num_classes"], help=f"default = {default_config['num_classes']}")
    parser.add_argument("--devices",                default=default_config["devices"], help=f'default = {default_config["devices"]}')
    parser.add_argument("--accelerator",            default=default_config["accelerator"], help=f'default = {default_config["accelerator"]}')
    parser.add_argument("--confidence-threshold", type=float, default=default_config["confidence_threshold"], help="used for filtering detections shown in W&B tables "
                                                                                                                    "and for dynamic cft during evaluation, "
                                                                                                                    f"default = {default_config['confidence_threshold']}")
    parser.add_argument("--no-wandb",               default=False, action='store_true', help="disable wandb logging during end of each validation batch")
    parser.add_argument("--not-mobile",             default=False, action='store_true', help="use the resnet50 backbone instead of mobilenet")
    parser.add_argument("--no-cft-eval",            default=False, action='store_true', help="disable confidence threshold during evaluation")
    parser.add_argument("--dynamic-cft-eval",       default=False, action='store_true', help="cft used in evaluation linearly increases "
                                                                                             "from 0 to specified threshold for every epoch")
    parser.add_argument("-q", "--quiet",            default=False, action='store_true', help="disables printing of arguments that are different from default")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    if not os.path.exists(default_config["train_split"]):
        fname = "drinks.tar.gz"
        url = f'https://github.com/DJSapit/Drinks-Dataset-Faster-RCNN-Sapit/releases/download/v0.1.0-alpha/{fname}'
        print(f'downloading drinks dataset from {url}')
        r = requests.get(url, allow_redirects=True)
        open('drinks.tar.gz', 'wb').write(r.content)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
        if os.path.exists(default_config["train_split"]):
            print("drinks dataset downloaded successfully")
        else:
            raise Exception("drinks dataset download failed")

    args = get_args()
    print(args)
    if not args.quiet:
        for arg in vars(args):
            if arg in default_config:
                if default_config[arg] != getattr(args, arg):
                    print(f'{arg}={getattr(args, arg)}')

    config = {
        "precision": args.precision,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "batch_size": args.batch_size,
        "dataset": default_config["dataset"],
        "train_split": default_config["train_split"],
        "test_split": default_config["test_split"],
        "is_mobile": not args.not_mobile}

    confidence_threshold = args.confidence_threshold
    cft_percent = f'{confidence_threshold*100}%'


class FasterRCNNModel(LightningModule):
    def __init__(self, num_classes=4, lr=0.005, batch_size=4):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        if args.not_mobile:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        self.model.cuda().eval()
        return self.model(x)

    # this is called during fit()
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum([loss for loss in loss_dict.values()])
        return {"loss": loss}

    # calls to self.log() are recorded in wandb
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_step=False, on_epoch=True)

    # this is called at the end of an epoch
    def test_step(self, batch, batch_idx):
        images, targets = batch
        with torch.no_grad():
            outs = self.model(images)

        if (args.max_epochs == 1) or (args.no_cft_eval):
            cft = 0
        elif not args.dynamic_cft_eval:
            cft = 0.6 if self.current_epoch > 0 else 0
        else:
            current_epoch = self.current_epoch if self.current_epoch <= (args.max_epochs - 1) else (args.max_epochs - 1)
            cft = confidence_threshold*current_epoch/(args.max_epochs - 1)

        iou = torch.stack([evaluate_iou(t, o, cft) for t, o in zip(targets, outs)]).mean()
        return {"iou": iou, "pred": outs}

    # this is called at the end of all epochs
    def test_epoch_end(self, outputs):
        avg_iou = torch.stack([o["iou"] for o in outputs]).mean()
        self.log("test_epoch_iou", avg_iou)
        return {"test_iou": avg_iou}

    def validation_step(self, batch, batch_idx):
       return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_iou = torch.stack([o["iou"] for o in outputs]).mean()
        self.log("val_epoch_iou", avg_iou)
        return {"val_iou": avg_iou}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0005,
        )
    
    # this is called after model instatiation to initiliaze the datasets and dataloaders
    def setup(self, stage=None):
        self.train_dataloader()
        self.test_dataloader()

    def train_dataloader(self):
        return DataLoader(DrinksDataset(config["train_split"]),
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=config['num_workers'],
                          pin_memory=config['pin_memory'],
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(DrinksDataset(config["test_split"]),
                         batch_size=config['batch_size'],
                         shuffle=False,
                         num_workers=config['num_workers'],
                         pin_memory=config['pin_memory'],
                         collate_fn=collate_fn)

    def val_dataloader(self):
        return self.test_dataloader()


class WandbCallback(Callback):
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        images, targets = batch
        predictions = outputs["pred"]
        columns = ['ground truth', 'prediction', f'prediction ; score > {cft_percent}']
        data = []
        for img, target, pred in zip(images, targets, predictions):
            image = T.ToPILImage()(img)
            ground_truth_boxes = target2boxes(target)
            pred_boxes = pred2boxes(pred)
            pred_thd_boxes = pred2boxes(pred, confidence_threshold)
            data.append([
                         wandb.Image(image, boxes=ground_truth_boxes),
                         wandb.Image(image, boxes=pred_boxes),
                         wandb.Image(image, boxes=pred_thd_boxes)
                        ])
        
        wandb_logger.log_table(
            key=f'FasterRCNN on Drinks Dataset Predictions batch_idx={batch_idx}',
            columns=columns,
            data=data)


if __name__ == "__main__":

    wandb.login()

    run = wandb.init(project=project_name, config=config)

    model = FasterRCNNModel(num_classes=args.num_classes,
                            lr=args.lr,
                            batch_size=args.batch_size)
    model.setup()

    wandb_logger = WandbLogger(project=project_name)
    
    if args.not_mobile:
        filename = 'fasterrcnn-{epoch}-{val_epoch_iou:.3f}'
    else:
        filename = 'fasterrcnn-mobile-{epoch}-{val_epoch_iou:.3f}'
    
    checkpoint_callback = ModelCheckpoint(save_top_k=default_config["save_top_k"],
                                        monitor="val_epoch_iou",
                                        mode="max",
                                        dirpath=default_config["checkpoint_dir"],
                                        filename=filename)


    trainer = Trainer(precision=args.precision,
                      accelerator=args.accelerator,
                      devices=args.devices,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger if not args.no_wandb else None,
                      callbacks=[checkpoint_callback, WandbCallback() if not args.no_wandb else None])

    model.cuda()
    trainer.fit(model)

    print("testing best checkpoint according to val_epoch_iou")
    print("best_model_path:", checkpoint_callback.best_model_path)
    model = FasterRCNNModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.cuda()
    trainer.test(model)

    test_data = DataLoader(DrinksDataset(default_config["test_split"]),
                         batch_size=default_config['batch_size'],
                         shuffle=False,
                         num_workers=default_config['num_workers'],
                         pin_memory=default_config['pin_memory'],
                         collate_fn=collate_fn)
    evaluate(model, test_data, device=torch.device("cuda"))

    wandb.finish()