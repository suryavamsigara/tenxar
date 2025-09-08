import tenxar
import numpy as np


def cross_entropy(logits: tenxar.tensor, y_labels: tenxar.tensor) -> tenxar.tensor:
    logits_max = logits.max(axis=1, keepdims=True)
    shifted_logits = logits - logits_max

    y_pred = softmax(shifted_logits)
    y_pred = y_pred[tenxar.arange(logits.shape[0]), y_labels]

    epsilon = 1e-9
    loss = -(y_pred + epsilon).log().mean()

    return loss

def softmax(logits: tenxar.tensor) -> tenxar.tensor:
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    return probs
    