import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

resnet = torchvision.models.resnet34(pretrained=True)
model = nn.Sequential(*list(resnet.children())[:5])
model2 = nn.Sequential(*list(resnet.children())[5:6])
model3 = nn.Sequential(*list(resnet.children())[6:7])
model4 = nn.Sequential(*list(resnet.children())[7:8])


class Custom_model(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.encoding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=2, bias=True, dilation=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=3, bias=True, dilation=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.level1 = nn.Sequential(*list(resnet.children())[5:6])
        self.level2 = nn.Sequential(*list(resnet.children())[6:7])
        self.level3 = nn.Sequential(*list(resnet.children())[7:8])

        # FPN layers
        self.FPN = torchvision.ops.FeaturePyramidNetwork([128, 256, 512], 128)

    def forward(self, x):
        encodings = self.encoding(x)
        level1 = self.level1(encodings)
        # print(level1.shape)
        level2 = self.level2(level1)
        # print(level2.shape)
        level3 = self.level3(level2)
        # print(level3.shape)
        x = OrderedDict()
        x["0"] = level1
        x["1"] = level2
        x["2"] = level3
        x = self.FPN(x)
        output = x["0"]
        return output


if __name__ == "__main__":
    model = Custom_model()
    # print(model)
    out = model(torch.randn(1, 3, 256, 256))
    print(out["0"].shape)
    print(out["1"].shape)
    print(out["2"].shape)
