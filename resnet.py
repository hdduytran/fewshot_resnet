from torch import nn
import torch

class ResBlock(nn.Module):
    def __init__(self,input_dim = 1, output_dim = 33, kernel_size_list = [4, 8, 16, 32, 64]) -> None:
        super().__init__()
        self.kernel_size_list = kernel_size_list
        self.convs1 = nn.ModuleList()
        for kernel_size in self.kernel_size_list:
            self.convs1.append(nn.Conv1d(input_dim,output_dim,kernel_size, padding='same', stride=1))
        generate_dim = len(self.kernel_size_list) * output_dim
        self.convs2 = nn.ModuleList()
        for kernel_size in self.kernel_size_list:
            self.convs2.append(nn.Conv1d(generate_dim,33,kernel_size, padding='same'))
        self.bn1 = nn.BatchNorm1d(generate_dim)
        self.bn2 = nn.BatchNorm1d(generate_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = input
        x = torch.cat([conv(x) for conv in self.convs1], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.cat([conv(x) for conv in self.convs2], dim=1)
        x = self.bn2(x)
        x = torch.add(x, input)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList()
        for _ in range(2):
            if len(self.resblocks) == 0:
                self.resblocks.append(ResBlock())
            else:
                prev_output_dim = self.resblocks[-1].convs1[-1].out_channels
                self.resblocks.append(ResBlock(input_dim=165))
        self.globalarveragepooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        x = input
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.globalarveragepooling(x)
        x = torch.flatten(x, 1)
        return x