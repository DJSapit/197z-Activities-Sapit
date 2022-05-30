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


Training
----------

```bash
python train.py
```
Dataset used for testing is automatically downloaded.

KWS Inference
----------

```bash
python kws-infer.py
```
Pretrained Model is automatically downloaded.


For `train.py` and `kws-infer.py`, add `-h` or `--help` to see a list of commands.

________
Code is heavily referenced from the following github links:
[KWS](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws_demo.ipynb) |
[Transformer](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/transformer/python/transformer_demo.ipynb) |
[kws-infer](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws-infer.py)
________
### [Watch Demo video here]()