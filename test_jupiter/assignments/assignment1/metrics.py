import numpy as np


def binary_classification_metrics(prediction: np.ndarray, ground_truth: np.ndarray):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # prediction = np.array()
    TP = np.sum(np.logical_and(prediction, ground_truth).astype(np.int32))
    FP = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)).astype(np.int32))
    FN = np.sum(np.logical_and(np.logical_not(prediction), ground_truth).astype(np.int32))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = np.sum((prediction == ground_truth).astype(np.int32)) / prediction.shape[0]
    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    # print(precision, recall, f1, accuracy)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return np.sum((prediction == ground_truth).astype(np.int32)) / prediction.shape[0]
