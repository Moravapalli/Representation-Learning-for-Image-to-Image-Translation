
import torch
import torch.nn as nn
#from torchsummary import summary
import config

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # Define convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, num_filters_last=128, n_layers=3):
        super(Discriminator, self).__init__()
        kw =3
        
        # Define the layers of the network 
        layers = [nn.Conv2d(256, num_filters_last, kw, 1, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last // num_filters_mult_last, num_filters_last // num_filters_mult, kw,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last // num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]
        layers.append(nn.Conv2d(num_filters_last // num_filters_mult, 1, kw, 1, 1))
        #layers.append(nn.Conv2d(32, 1, 4, 1, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor from the network.
        """
        return self.model(x)


# testing purpose
def test():
    x = torch.randn((5, 256, 16, 16))
    model = Discriminator()

    preds = model(x)
    #summary(model,(256,16,16))
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()
