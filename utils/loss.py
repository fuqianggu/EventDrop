import torch


def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def mse_loss_and_accuracy(prediction, target, num_classes=101):
    mse_loss = torch.nn.MSELoss()
    labels = torch.zeros(target.cpu().shape[0], num_classes).scatter_(1, target.cpu().view(-1, 1), 1)
    loss = mse_loss(prediction.cpu(), labels)
    _, pred_labels = prediction.cpu().max(1)
    accuracy = float(pred_labels.eq(target.cpu()).sum().item())
    return loss, accuracy
