import torch


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup(input, target, device, alpha=1.0):
    if alpha > 0:
        lambda_ = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lambda_ = 1

    batch_size = input.size(0)
    index = torch.randperm(batch_size).to(device=device)
    
    mixed_input = lambda_ * input + (1 - lambda_) * input[index, :]    
    labels_a, labels_b = target, target[index]

    return mixed_input, labels_a, labels_b, lambda_