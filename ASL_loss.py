import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.5, gamma_neg=3.0, eps=0.1, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_weight = targets * (1 - self.gamma_pos) + self.gamma_pos
        neg_weight = (1 - targets) * (1 - self.gamma_neg) + self.gamma_neg
        loss = -pos_weight * targets * torch.log(inputs + self.eps) - neg_weight * (1 - targets) * torch.log(
            1 - inputs + self.eps)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


if __name__ == '__main__':
    # Example usage
    num_classes = 4
    batch_size = 4

    # Generate random inputs and targets
    predicted = torch.tensor([0.9, 0.1, 0.8, 0.3], requires_grad=True)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    # Define the model and loss function
    model = nn.Linear(num_classes, num_classes)  # 两个线性层和ReLU和Sigmoid激活函数的简单多标签分类模型
    loss_fn = AsymmetricLoss()

    # Compute the loss
    outputs = model(predicted)
    loss = loss_fn(torch.sigmoid(outputs), target)
    print(loss)
