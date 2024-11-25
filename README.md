# MSAI Group 6
Zhengxiao Han, Zhengyang (Kris) Weng, Ben Benyamin

## Installation
```sh
pip install -r requirements.txt
```

## Dataset
In `data_generation`:
* `train.csv`, `valid.csv`, and `test.csv`.

## Quick Start
Terminal 1: Launch Tensorboard
```sh
tensorboard --logdir=runs
```
Terminal 2: Start Training
```sh
python preliminary/train.py
```
Then open [http://localhost:6006/](http://localhost:6006/) in your web browser. It will take a long time to load data before plotting in web browser.