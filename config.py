import multiprocessing as mp

project_name = "Object Detection on Drinks dataset using Faster R-CNN Model - Daniel Sapit"

class_labels = {0:'background', 1:'Water', 2:'Soda', 3:'Juice'}

pretrained_model_dir = "models/fasterrcnn_drinks_pretrained.ckpt"

default_config = {
        "precision": 16,
        "max_epochs": 5,
        "batch_size": 4,
        "lr": 0.001,
        "num_workers": 4,
        "num_classes": 4,
        "devices": "auto",
        "accelerator": "gpu",
        "confidence_threshold": 0.98,
        "pin_memory": True,
        "dataset": "drinks",
        "train_split": "drinks/labels_train.csv",
        "test_split": "drinks/labels_test.csv",
        "save_top_k": 1,
        "checkpoint_dir": "checkpoints"}