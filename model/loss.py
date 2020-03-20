import torch
import torch.nn as nn
from torchvision.models import resnet50
# from root path call and in the current path call is not the same
# if want to fix it. I think sys.path is the choice
from model.FAN import FAN
from torch.utils.model_zoo import load_url


class Generator_loss(nn.Module):
    def __init__(self):
        super(Generator_loss, self).__init__()
        self.fan_loss = FAN_loss()
        self.pixel_loss = Pixel_loss()
        self.perceptual_loss = Perceptual_loss()

    def forward(self, d_fake, data, target):
        adversarial_loss = -1 * torch.mean(d_fake)
        fan_loss = self.fan_loss(data, target.detach())
        pixel_loss = self.pixel_loss(data, target)
        perceptual_loss = self.perceptual_loss(data, target.detach())
        return pixel_loss + 0.1 * perceptual_loss + 0.005 * fan_loss + 0.01 * adversarial_loss


class FAN_loss(nn.Module):
    def __init__(self):
        super(FAN_loss, self).__init__()
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net
        self.criterion = nn.MSELoss()

    def forward(self, data, target):
        # data = self.FAN_net(data)
        # target = self.FAN_net(target)
        # print(data[0].size())
        # print(target[0].size())
        # exit()
        return self.criterion(self.FAN_net(data)[0], self.FAN_net(target)[0])

class Pixel_loss(nn.Module):
    def __init__(self):
        super(Pixel_loss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, data, target):
        return self.criterion(data, target)


class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.criterion = nn.MSELoss()
        resnet = resnet50(pretrained=True)

        self.B1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        for p in self.B1.parameters():
            p.requires_grad = False

        self.B2 = resnet.layer2
        for p in self.B2.parameters():
            p.requires_grad = False

        self.B3 = resnet.layer3
        for p in self.B3.parameters():
            p.requires_grad = False


    def forward(self, data, target):
        data, target = self.B1(data), self.B1(target)
        B1_loss = self.criterion(data, target)
        data, target = self.B2(data), self.B2(target)
        B2_loss = self.criterion(data, target)
        data, target = self.B3(data), self.B3(data)
        B3_loss = self.criterion(data, target)
        return B1_loss + B2_loss + B3_loss

if __name__ == "__main__":
    net = Generator_loss()
    d_fake = torch.randn(2, 1)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    print(net(d_fake, x, y))
    exit()


    net = resnet_loss()
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    loss = net(x, y)
    print(loss)
