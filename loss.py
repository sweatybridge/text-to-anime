import torch
from torch import nn


class AdaptiveWingLoss(nn.Module):
    """
    https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py
    """

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        """

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        # torch.log and math.log is e based
        loss1 = self.omega * torch.log(
            1 + torch.pow(delta_y1 / self.omega, self.alpha - y1)
        )
        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)))
            * (self.alpha - y2)
            * (torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)
        )
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse(mel_out, mel_target)
        post_loss = self.mse(mel_out_postnet, mel_target)
        gate_loss = self.bce(gate_out, gate_target)
        return mel_loss + post_loss + gate_loss


class TextLandmarkLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xyz_loss = nn.SmoothL1Loss()
        # self.gate_loss = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        xyz_target = targets[0]
        xyz_target.requires_grad = False

        xyz_out, xyz_out_postnet, _, _ = model_output
        xyz_loss = self.xyz_loss(xyz_out, xyz_target)
        post_loss = self.xyz_loss(xyz_out_postnet, xyz_target)

        return xyz_loss + post_loss


if __name__ == "__main__":
    loss_func = AdaptiveWingLoss()
    y = torch.ones(2, 68, 64, 64)
    y_hat = torch.zeros(2, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)
