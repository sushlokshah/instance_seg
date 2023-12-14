import torch
import torchvision
from torchvision.transforms import transforms
from torchview import draw_graph
from collections import OrderedDict

x = OrderedDict()
x["feat0"] = torch.rand(1, 10, 64, 64)
x["feat2"] = torch.rand(1, 20, 16, 16)
x["feat3"] = torch.rand(1, 30, 8, 8)

m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
m.summary()
# model_graph = draw_graph(m, input_size=(1, 10, 20, 30), expand_nested=True)
# model_graph.visual_graph
