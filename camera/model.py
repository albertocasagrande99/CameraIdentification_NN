from .scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, log_loss
from torch.autograd import Variable
from torchvision.models import resnet50
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # Import mlines to customize the legend marker size
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

NUM_CLASSES = 8

colors_per_class = {
    '0' : 'yellow',
    '1' : 'red',
    '2' : 'orchid',
    '3' : 'darkmagenta',
    '4' : 'deepskyblue',
    '5' : 'turquoise',
    '6' : 'green',
    '7' : 'black'
}

label_to_name = {
    '0' : 'Apple_iPadmini5',
    '1' : 'Apple_iPhone13',
    '2' : 'Huawei_P20Lite',
    '3' : 'Motorola_MotoG6Play',
    '4' : 'Samsung_GalaxyA71',
    '5' : 'Samsung_TabA',
    '6' : 'Samsung_TabS5e',
    '7' : 'Sony_XperiaZ5',
}

class SerializableModule(nn.Module):
    def __init__(self):
        super(SerializableModule, self).__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class Model(SerializableModule):
    def __init__(self, weights_path=None):
        super(Model, self).__init__()

        model = resnet50()
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)

        num_features = model.fc.in_features
        model.fc = nn.Dropout(0.0)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

        self._model = model

    def forward(self, x):
        x = self._model(x)
        penultimate_layer_features = self.bottleneck(x)
        x = self.fc(penultimate_layer_features)
        return x, penultimate_layer_features


class CameraModel(object):
    def __init__(self, model=None, resume_training=0):
        if model is None:
            model = Model()
        self._torch_single_model = model
        if resume_training == 1:
            self._torch_single_model.load_state_dict(torch.load("/content/drive/MyDrive/Magistrale/Tirocinio Bergen/Dataset_NN/modelRear.pth"))
        self._torch_model = nn.DataParallel(self._torch_single_model).cuda()

        self._optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=0.0001)
        self._scheduler = ReduceLROnPlateau(self._optimizer, factor=0.5, patience=3,
                                            min_lr=1e-6, epsilon=1e-5, verbose=1, mode='min')
        self._optimizer.zero_grad()
        self._criterion = nn.CrossEntropyLoss()

    def scheduler_step(self, loss, epoch):
        self._scheduler.step(loss, epoch)

    def enable_train_mode(self):
        self._torch_model.train()

    def enable_predict_mode(self):
        self._torch_model.eval()

    def train_on_batch(self, X, y):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        X = Variable(X, requires_grad=False)
        y = Variable(y, requires_grad=False)

        y_pred, _ = self._torch_model(X)

        loss = self._criterion(y_pred, y)
        loss.backward()

        self._optimizer.step()
        self._optimizer.zero_grad()

        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred.cpu().data.numpy()

    def predict_on_batch(self, X):
        X = X.cuda(non_blocking=True)
        with torch.no_grad():
            X = Variable(X, requires_grad=False)
            y_pred, penultimate_layer_features = self._torch_model(X)
        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred.cpu().data.numpy(), penultimate_layer_features

    def fit_generator(self, generator):
        self.enable_train_mode()
        mean_loss = None
        mean_accuracy = None
        start_time = time.time()
        for step_no, (X, y) in enumerate(generator):
            y_pred = self.train_on_batch(X, y)
            y = y.cpu().numpy()

            accuracy = accuracy_score(y, y_pred.argmax(axis=-1))
            loss = log_loss(y, y_pred, eps=1e-6, labels=list(range(8)))

            if mean_loss is None:
                mean_loss = loss

            if mean_accuracy is None:
                mean_accuracy = accuracy

            mean_loss = 0.9 * mean_loss + 0.1 * loss
            mean_accuracy = 0.9 * mean_accuracy + 0.1 * accuracy

            cur_time = time.time() - start_time
            print("[{3} s] Train step {0}. Loss {1}. Accuracy {2}".format(step_no, mean_loss, mean_accuracy, cur_time))

    def predict_generator(self, generator):
        self.enable_predict_mode()
        result = []
        start_time = time.time()
        # Initialize lists to store features and labels
        features = None
        labels = []
        # Loop through your data and extract features
        with torch.no_grad():
            for images, targets in generator:
                # Pass the images through the CameraModel to get penultimate layer features
                labels.extend(targets)

        for step_no, X in enumerate(generator):
            if isinstance(X, (tuple, list)):
                X = X[0]

            y_pred, penultimate_layer_features = self.predict_on_batch(X)
            current_features = penultimate_layer_features.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features
            result.append(y_pred)
            print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=3, learning_rate='auto', init='pca')
        tsne_results = tsne.fit_transform(features)
        labels = [label.item() for label in labels]
        plt.style.use('seaborn')
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 0.7
        # Create a larger figure
        fig = plt.figure(figsize=(7, 5))  # Adjust the values (width, height) as desired
        ax = fig.add_subplot(111, projection='3d')
        for label in label_to_name:
            indices = [i for i, lbl in enumerate(labels) if lbl == int(label)]  # Assuming 'labels' contains class labels
            color = colors_per_class[label]
            ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1], tsne_results[indices, 2], label=label_to_name[label], c=color, s=10)
        # Move the legend to the right of the plot
        legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # Adjust the bbox_to_anchor values as needed
        # Increase the marker size in the legend
        # Increase the marker size in the legend
        for handle in legend.legendHandles:
            if isinstance(handle, matplotlib.lines.Line2D):  # Check if it's a scatter plot marker
                handle.set_markersize(14)  # Adjust the marker size as needed
        plt.subplots_adjust(right=0.70)
        plt.title('t-SNE Visualization of Class Representations')
        plt.savefig('t-SNE.pdf', format="pdf")
        return np.concatenate(result)

    def save(self, filename):
        self._torch_single_model.save(filename)

    @staticmethod
    def load(filename):
        model = Model()
        model.load(filename)
        return CameraModel(model=model)
