import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, mode):
        super(Loss, self).__init__()
        self.mode = mode

    def forward(self, output, target):
        if self.mode == 'CE':
            metric = CE()
        elif self.mode == 'Focal':
            metric = Focal()
        else:
            print('The chosen loss function is not provided!!!! We use CE Loss instead!!!')
            metric = CE()
        loss = metric(output, target)
        return loss


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

    def forward(self, output, target):
        outputF = torch.log(output + 1e-10)
        outputB = torch.log(1 - output + 1e-10)
        loss = -target * outputF - (1 - target) * outputB
        return loss.sum(dim=1)


class Focal(nn.Module):
    def __init__(self):
        super(Focal, self).__init__()

    def forward(self, output, target):
        gamma = 2.5
        outputF = torch.log(output + 1e-10)
        outputB = torch.log(1 - output + 1e-10)
        lossF = -torch.pow(1-output, gamma) * target * outputF
        lossB = -torch.pow(output, gamma) * (1 - target) * outputB
        loss = lossF + lossB
        return loss.sum(dim=1)