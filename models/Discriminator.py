from .BasicModule import BasicModule
from torch import nn


class Discriminator(BasicModule):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = opt.discriminator_feature_maps
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.linear = nn.Linear(ndf * 8 * 4 * 4, 1)

    def forward(self, input_data):
        temp = self.main(input_data).view(-1, 64 * 8 * 4 * 4)
        return self.linear(temp)

