import fiftyone as fo
import fiftyone.zoo as foz
import os

dataset_name = "cityscapes"
path = "/home/awi-docker/video_summarization/instance_seg/dataset/" + dataset_name
if not os.path.exists(path):
    os.makedirs(path)

dataset = foz.load_zoo_dataset(
    "coco-2017",
    dataset_dir=path,
    dataset_name=dataset_name,
)
session = fo.launch_app(dataset)
session.wait()
