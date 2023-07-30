"""
this file is used to create a small dataset from the coco-2017 dataset (we only use it to create a small pickle dataset for testing purposes)
"""
import torchvision
import textwrap
import fiftyone.zoo as foz
import torchvision.transforms.v2 as transforms
from collections import defaultdict
import PIL.Image
import torch
import pickle

path2data="./data/validation/data"
path2json="./data/raw/instances_val2017.json"
label_map_dataset={
    0:"0",
    2:"bicycle",
    4:"motorcycle",
    10:"traffic light",
    13:"stop sign",
    75:"remote",
    77:"cell phone",
}


def filter_dataset(dataset):
    filtered_dataset = []
    for image, targets in dataset:
        if "labels" not in targets:
            continue
        new_targets = {"labels":[],"boxes":[]}
        for l,bb in zip(targets["labels"],targets["boxes"]):
            l = int(l)
            if l in labels_map:
                new_targets["labels"].append(l)
                new_targets["boxes"].append(bb)
        if len(new_targets["labels"]) > 0:
            filtered_dataset.append((image,new_targets))
        

if __name__=="__main__":
    
    foz.download_zoo_dataset("coco-2017",dataset_dir="./data",split="validation"
                                )
    transform = transforms.Compose(
        [
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    coco_dataset = torchvision.datasets.CocoDetection(root = path2data,
                                    annFile = path2json,transform=transform)
    dataset=torchvision.datasets.wrap_dataset_for_transforms_v2(coco_dataset)
    filtered_dataset = filter_dataset(dataset)
    with open("./small_dataset.pkl","wb") as f:
        pickle.dump(filtered_dataset,f)