# Camera Model Identification through neural networks

This repo contains code for training and predicting single Resnet50 model.

# Run training

Next, download pretrained weights from https://download.pytorch.org/models/resnet50-19c8e357.pth

Next, call
`python train.py --train_files train_files --val_files val_files --pretrained_weights_path resnet50-19c8e357.pth --batch_size 128 --model_save_path model.pth`

You will need some time to train a model. It takes ~4 hours on a single Tesla M40.

# Run prediction

Just call

`python predict.py --test_files test_files --batch_size 128 --model_path model.pth --submit_path submit.csv`

In the finish, there will be a file submit.csv which you will be able to submit on Kaggle.
