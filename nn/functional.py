from ..tensor import Tensor, arange
import numpy as np

def cross_entropy(logits: Tensor, y_labels: Tensor) -> Tensor:
    if y_labels.data.ndim > 1:
        y_labels = y_labels.squeeze()
    print(y_labels)
    logits_max = logits.max(axis=1, keepdims=True)
    shifted_logits = logits - logits_max

    y_pred = softmax(shifted_logits)
    y_pred = y_pred[arange(logits.shape[0]), y_labels]

    epsilon = 1e-9
    loss = -(y_pred + epsilon).log().mean()

    return loss

def softmax(logits: Tensor) -> Tensor:
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    return probs
    
def binary_cross_entropy(logits: Tensor, y_labels: Tensor) -> Tensor:
    if logits.shape != y_labels.shape:
        raise ValueError(f"Shapes should be the same: {logits.shape}, {y_labels.shape}")
    
    probs = logits.sigmoid()

    epsilon = 1e-9

    loss = -(y_labels * (probs + epsilon).log() + (1 - y_labels) * (1 - probs + epsilon).log()).mean()

    return loss
