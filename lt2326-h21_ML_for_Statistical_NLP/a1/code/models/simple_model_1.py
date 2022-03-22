import torch
import torch.nn as nn
from torch import Tensor


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        kwargs = dict(kernel_size=8, stride=4, padding=2)  # change size by 4

        self.downsampling_1 = nn.Sequential(nn.Conv2d(3, 32, **kwargs),
                                            nn.ReLU())

        self.downsampling_2 = nn.Sequential(nn.Conv2d(32, 128, **kwargs),
                                            nn.ReLU())

        self.downsampling_3 = nn.Sequential(nn.Conv2d(128, 256, **kwargs),
                                            nn.ReLU())

        self.upsampling_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, **kwargs), nn.ReLU())

        self.upsampling_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, **kwargs), nn.ReLU())

        self.out = nn.Sequential(nn.ConvTranspose2d(32, 4, **kwargs),
                                 nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsampling_1(x)
        x = self.downsampling_2(x)
        x = self.downsampling_3(x)
        x = self.upsampling_1(x)
        x = self.upsampling_2(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:2")
    criterion = nn.CrossEntropyLoss().to(device)
    b = 18

    xin = torch.randn((b, 3, 2048, 2048)).to(device)
    test_model = Model_1().to(device)
    xout = test_model(xin)
    print(f"input size:  {xin.size()}")
    print(f"output size: {xout.size()}")

    target = torch.empty(b, 2048, 2048, dtype=torch.long).random_(4).to(device)
    print(f"target size: {target.size()}")

    loss = criterion(xout, target)
    loss.backward()
