# Drinks Dataset Object Detection using FasterRCNN
197z Deep Learning Assignment 2 by Daniel Sapit

### Model Paper
* [Arxiv](https://arxiv.org/abs/1506.01497)
### Model Code Reference
* [FASTERRCNN_RESNET50_FPN](https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
* [FASTERRCNN_MOBILENET_V3_LARGE_320_FPN](https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)

________
Installation
----------

1. Clone repo

    ```bash
    git clone https://github.com/DJSapit/Drinks-Dataset-Faster-RCNN-Sapit.git
    cd Drinks-Dataset-Faster-RCNN-Sapit
    ```

2. Install requirements

    ```bash
    pip install -r requirements.txt
    ```
Testing
----------

```bash
python test.py
```
Pretrained Model and Dataset used for testing are automatically downloaded.

Training
----------

```bash
python train.py
```
Dataset used for testing is automatically downloaded.

Demo
----------

```bash
python demo.py
```
Pretrained Model used for webcam inference is automatically downloaded.


For `train.py` and `demo.py`, add `-h` or `--help` to see a list of commands.

You can edit the default values of the parameters used for testing and training in `config.py`.

________
### [Watch Demo video here](https://drive.google.com/file/d/1n37MeEjphP0xdNksy18eRh_VVImC4YTy/view?usp=sharing)