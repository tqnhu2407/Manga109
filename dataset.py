import numpy as np
import pandas as pd
from PIL import Image
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_df, classes, height, width, transform):
        self.images = images_df
        self.images_paths = images_df["path"].to_list()
        self.annotations = images_df["annotation"]
        self.classes = classes
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
        image_resized = image.resize((self.height, self.width))
        
        boxes = []
        labels = []
        image_width, image_height = image.size

        for annotation_type in self.classes:
            rois = self.annotations[idx][annotation_type]
            for roi in rois:
                labels.append(self.classes.index(annotation_type))
                xmin = roi["@xmin"]
                ymin = roi["@ymin"]
                xmax = roi["@xmax"]
                ymax = roi["@ymax"]

                # resize bounding box according to the desired size
                xmin_final = (xmin/image_width) * self.width
                xmax_final = (xmax/image_width) * self.width
                ymin_final = (ymin/image_height) * self.height
                yamx_final = (ymax/image_height) * self.height
                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        image_resized = self.transform(image_resized)

        return image_resized, target

    def __len__(self):
        return len(self.images)