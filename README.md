# Camera Model Identification through CNN

This repo is based on [this work](https://github.com/PavelOstyakov/camera_identification/), and contains code for training a Resnet50 model for camera model identification. 📈

## Dataset 📁
The *DivNoise* dataset can be downloaded at [https://divnoise.fotoverifier.eu/](https://divnoise.fotoverifier.eu/). Part 1 refers to smartphones, tablets and webcams, while the remaining parts contain the Canon cameras. 
To properly train and test the network, the images must be rearranged into train, validation and test splits for each camera.

## Training 🏃

- Split the dataset into train, validation and test sets.
- Next, download pretrained weights from https://download.pytorch.org/models/resnet50-19c8e357.pth
- Specify the paths to the images in the `train_files`, `val_files` and `test_files`.
- Finally, call:
```
python train.py --train_files train_files --val_files val_files --pretrained_weights_path resnet50-19c8e357.pth --batch_size 128 --model_save_path model.pth --resume 0
```

To train a model, it takes ~4 hours on a single Tesla M40.

## Testing 🚀

Just call:

```
python predict.py --test_files test_files --batch_size 128 --model_path model.pth --submit_path submit.csv
```
