import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# from torchvision.datasets import Cityscapes
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import os
from PIL import Image
import json


class Cityscapes(data.Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transforms (callable, optional):
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )

    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass(
            "rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)
        ),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass(
            "building", 11, 2, "construction", 2, False, False, (70, 70, 70)
        ),
        CityscapesClass(
            "wall", 12, 3, "construction", 2, False, False, (102, 102, 156)
        ),
        CityscapesClass(
            "fence", 13, 4, "construction", 2, False, False, (190, 153, 153)
        ),
        CityscapesClass(
            "guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)
        ),
        CityscapesClass(
            "bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)
        ),
        CityscapesClass(
            "tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)
        ),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass(
            "polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)
        ),
        CityscapesClass(
            "traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)
        ),
        CityscapesClass(
            "traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)
        ),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass(
            "license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)
        ),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        transforms: Optional[Callable] = None,
        random_crop=False,
        random_flip=0,
        random_rotate=0,
        random_scale=0,
    ) -> None:
        super(Cityscapes, self).__init__()
        self.root = root
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.transforms = transforms
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_scale = random_scale

        if not isinstance(target_type, list):
            self.target_type = [target_type]

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0],
                        self._get_target_suffix(self.mode, t),
                    )
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")
        image = image.resize((1024, 512), Image.BICUBIC)
        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)
        target = targets
        # target = list(targets) if len(targets) > 1 else targets[0]

        if self.random_scale > 0:
            scale = np.random.uniform(1 - self.random_scale, 1 + self.random_scale)
            w, h = image.size
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            for i, t in enumerate(target):
                target[i] = t.resize((int(w * scale), int(h * scale)))

        if self.random_crop:
            # random crop size 2/3H * 2/3W
            w, h = image.size
            th, tw = int(h / 2), int(w / 2)
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)
            image = image.crop((x1, y1, x1 + tw, y1 + th))
            for i, t in enumerate(target):
                target[i] = t.crop((x1, y1, x1 + tw, y1 + th))

        if self.random_flip > 0:
            if np.random.rand() < self.random_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                for i, t in enumerate(target):
                    target[i] = t.transpose(Image.FLIP_LEFT_RIGHT)

        if self.random_rotate > 0:
            if np.random.rand() < 0.4:
                angle = np.random.randint(-self.random_rotate, self.random_rotate)
                image = image.rotate(angle)
                for i, t in enumerate(target):
                    target[i] = t.rotate(angle)

        for i, t in enumerate(target):
            width, height = t.size
            target[i] = target[i].resize((width // 2, height // 2), Image.NEAREST)

        if self.transforms is not None:
            image = self.transforms(image)
            transformed_target = []
            for t in target:
                t = self.transforms(t)
                transformed_target.append(t)
            target = (
                tuple(transformed_target)
                if len(transformed_target) > 1
                else transformed_target[0]
            )

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"


# define custom transform class to apply same transform to both image and target


if __name__ == "__main__":
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = Cityscapes(
        "/home/awi-docker/video_summarization/instance_seg/dataset/cityscapes/",
        split="train",
        mode="fine",
        target_type=["instance", "color"],
        # transforms=train_transform,
        random_crop=True,
        random_flip=0.5,
        # random_rotate=10,
    )

    print(len(dataset))
    img_org, (inst, col) = dataset[0]
    inst = np.array(inst)
    print(inst.max(), inst.min())
    # print(img_org.shape, inst.shape, col.shape)
    vis_path = "/home/awi-docker/video_summarization/instance_seg/dataset/vis/"
    cv2.imwrite(vis_path + "org.png", np.array(img_org))
    # # # get unique values (instances)
    unique = np.unique(inst)
    print(len(unique))
    # for i in unique:
    #     mask = inst == i
    #     mask = mask.astype(np.uint8)
    #     mask = mask * 255
    #     cv2.imwrite(vis_path + str(i) + ".png", mask)
    #     print(i)

    cv2.imwrite(vis_path + "col.png", np.array(col)[:, :, :3])
    # # print(poly)
