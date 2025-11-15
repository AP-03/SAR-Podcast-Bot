import torch

def multilabel_f1(preds, targets, threshold=0.5, eps=1e-7):
    """
    preds:   [N, C] probabilities in [0,1]
    targets: [N, C] 0/1
    returns macro-F1 over C classes
    """
    preds_bin = (preds >= threshold).float()
    tp = (preds_bin * targets).sum(dim=0)
    fp = (preds_bin * (1 - targets)).sum(dim=0)
    fn = ((1 - preds_bin) * targets).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return f1.mean().item(), f1.cpu().numpy()
