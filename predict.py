from camera.augmentation import Augmentator
from camera.dataset import CameraDataset
from camera.model import CameraModel
from camera.postprocessing import generate_submit
from camera.train_utils import predict_test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import argparse

PROCESS_COUNT = 2


def main():
    parser = argparse.ArgumentParser("Camera Kaggle competition PREDICT")
    parser.add_argument("--test_files", required=True)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--submit_path", required=True)

    args = parser.parse_args()

    test_augmentator = Augmentator(in_train_mode=False)

    test_dataset = CameraDataset(args.test_files, test_augmentator, expand_dataset=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=PROCESS_COUNT)

    camera_model = CameraModel.load(args.model_path)
    predicts = predict_test(camera_model, test_loader)

    test_files = open(args.test_files).read().split("\n")[:-1]
    predicted_labels = generate_submit(predicts, test_files, args.submit_path)

    true_labels = test_dataset._labels
    true_labels = np.array(true_labels)
    labels = ['Apple_iPadmini5_F', 'Apple_iPadmini5_R', 'Apple_iPhone13_F', 'Apple_iPhone13_R', 'Huawei_P20Lite_F', 'Huawei_P20Lite_R', 'Motorola_MotoG6Play_F', 'Motorola_MotoG6Play_R', 'Samsung_GalaxyA71_F', 'Samsung_GalaxyA71_R', 'Samsung_GalaxyTabA_F', 'Samsung_GalaxyTabA_R', 'Samsung_GalaxyTabS5e_F', 'Samsung_GalaxyTabS5e_R', 'Sony_XperiaZ5_F', 'Sony_XperiaZ5_R']
    cm = confusion_matrix(true_labels, predicted_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 12})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,15))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax, values_format='.2f')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("cm.pdf", format="pdf", pad_inches=5)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    main()
