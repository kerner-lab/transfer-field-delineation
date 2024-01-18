import torch
import torch.nn.functional as F

def calculate_metrics(outputs, targets):
    # Convert outputs and targets to binary predictions
    predictions = torch.round(torch.sigmoid(outputs))
    targets = targets.float()

    # Calculate true positives, false positives, and false negatives
    true_positives = torch.sum(predictions * targets)
    false_positives = torch.sum(predictions) - true_positives
    false_negatives = torch.sum(targets) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Calculate accuracy
    accuracy = torch.mean((predictions == targets).float())

    # Calculate IoU (Intersection over Union)
    intersection = torch.sum(predictions * targets)
    union = torch.sum(predictions) + torch.sum(targets) - intersection
    iou = intersection / (union + 1e-7)

    return f1_score, accuracy, iou, precision

