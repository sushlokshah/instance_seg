import fiftyone as fo
import fiftyone.zoo as foz

# The path to the source files that you manually downloaded
source_dir = "/home/awi-docker/video_summarization/instance_seg/dataset/cityscapes"

dataset = foz.load_zoo_dataset(
    "cityscapes",
    source_dir=source_dir,
    dataset_dir="/home/awi-docker/video_summarization/instance_seg/dataset/cityscapes",
)
