import torch
import torch.nn as nn
from torch import Tensor


class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        kwargs = dict(kernel_size=8, stride=4, padding=2)  # change size by 4
        c = 32  # number of channels

        self.downsampling_1 = nn.Sequential(nn.Conv2d(3, c, **kwargs),
                                            nn.ReLU())

        self.downsampling_2 = nn.Sequential(nn.Conv2d(c, c * 2, **kwargs),
                                            nn.ReLU())

        self.downsampling_3 = nn.Sequential(nn.Conv2d(c * 2, c * 8, **kwargs),
                                            nn.ReLU())

        self.downsampling_4 = nn.Sequential(nn.Conv2d(c * 8, c * 16, **kwargs),
                                            nn.ReLU())

        self.upsampling_1 = nn.Sequential(
            nn.ConvTranspose2d(c * 16, c * 8, **kwargs), nn.ReLU())

        self.upsampling_2 = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 2, **kwargs), nn.ReLU())

        self.upsampling_3 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, **kwargs), nn.ReLU())

        self.out = nn.Sequential(nn.ConvTranspose2d(c, 4, **kwargs), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.downsampling_1(x)    # H/4, W/4
        x = self.downsampling_2(x1)    # H/16, W/16
        x2 = self.downsampling_3(x)    # H/64, W/64
        x = self.downsampling_4(x2)    # H/256, W/256
        x = self.upsampling_1(x)       # H/64, W/64
        x = self.upsampling_2(x + x2)  # H/16, W/16
        x = self.upsampling_3(x)       # H/4, W/4
        x = self.out(x + x1)           # H, W
        return x


if __name__ == "__main__":
    device = torch.device("cuda:2")
    criterion = nn.CrossEntropyLoss().to(device)
    b = 18

    xin = torch.randn((b, 3, 2048, 2048)).to(device)
    test_model = Model_2().to(device)
    xout = test_model(xin)
    print(f"input size:  {xin.size()}")
    print(f"output size: {xout.size()}")

    target = torch.empty(b, 2048, 2048, dtype=torch.long).random_(4).to(device)
    print(f"target size: {target.size()}")

    loss = criterion(xout, target)
    loss.backward()
