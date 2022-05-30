# Keyword Spotting Transformer
197z Deep Learning Assignment 3 by Daniel Sapit

________
Installation
----------

1. Clone repo

    ```bash
    git clone https://github.com/DJSapit/197z-Activities-Sapit.git
    cd 197z-Activities-Sapit
    ```

2. Install requirements

    ```bash
    pip install -r requirements.txt
    ```
Sidenote: some packages may not work properly when run on a conda environment where the libraries are installed using conda. 
Some listed packages may came from the previous activity, Drinks Dataset, and may not need to be installed.


Training
----------

```bash
python train.py
```
Highest test accuracy achieved was 93% with a 651k parameter model.
 
KWS Inference
----------

```bash
python kws-infer.py
```
Pretrained Model is automatically downloaded. Use '--button' flag to enable manual activation of recording.
Press 'q' or 'esc' to quit. Tap any other key to activate 1 second recording.


For `train.py` and `kws-infer.py`, add `-h` or `--help` to see a list of commands.

________
Code is heavily referenced from the following github links:
[KWS](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws_demo.ipynb) |
[Transformer](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/transformer/python/transformer_demo.ipynb) |
[kws-infer](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws-infer.py)
________
### [Watch Demo video here](https://drive.google.com/file/d/10GhWJ28dCI0weOeIUKT0QZQpLAuTlRZT/view?usp=sharing)