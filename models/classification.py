import torch.nn as nn
import torch


# class Segment(nn.Module):
#     def __init__(self, in_channels, classes, kernel_size, stride, padding):
#         super(Segment, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU()
#         self.
