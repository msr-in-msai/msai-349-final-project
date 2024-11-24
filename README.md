# MSAI Group 6
Zhengxiao Han, Zhengyang (Kris) Weng, Ben Benyamin

## Installation
```sh
pip install -r requirements.txt
```

## Dataset
`train.csv`: [https://drive.google.com/file/d/1xVAWzAU4cCpIHK2FUVID6CIGrfInXvnQ/view?usp=sharing](https://drive.google.com/file/d/1xVAWzAU4cCpIHK2FUVID6CIGrfInXvnQ/view?usp=sharing)

`valid.csv`: [https://drive.google.com/file/d/1LHz2exMdY1rmRTd3IUDu_SLaP6vrslPM/view?usp=sharing](https://drive.google.com/file/d/1LHz2exMdY1rmRTd3IUDu_SLaP6vrslPM/view?usp=sharing)

`test.csv`: [https://drive.google.com/file/d/1BGdonjtmS5QknrANZs3PCI-AtbFCmYgg/view?usp=drive_link](https://drive.google.com/file/d/1BGdonjtmS5QknrANZs3PCI-AtbFCmYgg/view?usp=drive_link)

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