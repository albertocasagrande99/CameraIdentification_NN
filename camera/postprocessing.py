from collections import Counter

import numpy as np

import copy

TTA_COUNT = 1

CLASSES = [
    "Huawei_P20Lite_Frontal",
    "Huawei_P20Lite_Rear"
    "Apple_iPadmini5_Frontal",
    "Apple_iPadmini5_Rear", 
    "Apple_iPhone13_Frontal", 
    "Apple_iPhone13_Rear", 
    "Huawei_P20Lite_Frontal", 
    "Huawei_P20Lite_Rear", 
    "Motorola_MotoG6Play_Frontal", 
    "Motorola_MotoG6Play_Rear", 
    "Samsung_GalaxyA71_Frontal", 
    "Samsung_GalaxyA71_Rear", 
    "Samsung_GalaxyTabA_Frontal", 
    "Samsung_GalaxyTabA_Rear", 
    "Samsung_GalaxyTabS5e_Frontal", 
    "Samsung_GalaxyTabS5e_Rear", 
    "Sony_XperiaZ5_Frontal", 
    "Sony_XperiaZ5_Rear"
]


def _geometric_mean(list_preds):
    result = np.ones((291, 2))
    for predict in list_preds:
        result *= predict
    result **= 1. / len(list_preds)
    return result


def _get_predicts(predicts, coefficients):
    predicts = copy.deepcopy(predicts)
    for i in range(len(coefficients)):
        predicts[:, i] *= coefficients[i]

    return predicts


def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = predicts.argmax(axis=-1)
    counter = Counter(labels)
    return labels, counter


def _compute_score_with_coefficients(predicts, coefficients):
    _, counter = _get_labels_distribution(predicts, coefficients)

    score = 0.
    for label in range(2):
        score += min(100. * counter[label] / len(predicts), 2)
    return score


def _find_best_coefficients(predicts, alpha=0.001, iterations=10000):
    coefficients = [1] * 2

    best_coefficients = coefficients[:]
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in range(iterations):
        _, counter = _get_labels_distribution(predicts, coefficients)
        labels_distribution = map(lambda x: x[1], sorted(counter.items()))
        label = np.argmax(labels_distribution)
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)
        if score > best_score:
            best_score = score
            best_coefficients = coefficients[:]

    return best_coefficients


def generate_submit(predictions, files, output_file):
    results = []
    for i in range(TTA_COUNT):
        results.append(predictions[i::TTA_COUNT])
    predictions = _geometric_mean(results)

    coefficients = _find_best_coefficients(predictions)
    labels_pred, _ = _get_labels_distribution(predictions, coefficients)
    labels = map(lambda label: CLASSES[label], labels_pred)
    filenames = map(lambda x: x[x.find("/") + 1:], files)
    result = zip(filenames, labels)

    with open(output_file, "w") as f:
        f.write("fname,camera\n")
        for filename, label in result:
            f.write(filename)
            f.write(",")
            f.write(label)
            f.write("\n")
    return labels_pred
