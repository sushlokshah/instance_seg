import fiftyone as fo
import fiftyone.zoo as foz

# The path to the source files that you manually downloaded
source_dir = (
    "/home/awi-docker/video_summarization/instance_seg/dataset/cityscapes_original"
)

dataset = foz.load_zoo_dataset(
    "cityscapes",
    split="validation",
    source_dir=source_dir,
)

session = fo.launch_app(dataset)
session.wait()
