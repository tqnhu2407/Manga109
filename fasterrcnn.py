import os
from tqdm.auto import tqdm
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import CustomDataset
from utils import *

def main():
    manga109_root_dir = "./Manga109_released_2023_12_07"
    parser = manga109api.Parser(root_dir=manga109_root_dir)

    CLASSES = ["body", "face", "frame", "text"]
    num_classes = len(CLASSES)
    BATCH_SIZE = 8
    NUM_WORKERS = 0

    images = load_all_images(parser, CLASSES)

    train_images, val_images = train_test_split(images, shuffle=True, test_size=0.2, random_state=42)
    df_train = pd.DataFrame(train_images, columns=["path", "annotation"])
    df_val = pd.DataFrame(val_images, columns=["path", "annotation"])

    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    train_dataset = CustomDataset(df_train, CLASSES, 512, 512, transform)
    val_dataset = CustomDataset(df_val, CLASSES, 512, 512, transform)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    def collate_fn(batch):
        """
        To handle the data loading as different images may have different number 
        of objects and to handle varying size tensors as well.
        """
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\nSTART LOADING MODEL\n")
    start_time = time.time()
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)
    end_time = time.time()
    print("\n\nFINISH LOADING MODEL\n")
    print(f"MODEL LOADING TIME = {end_time - start_time} seconds\n")
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n,p in model.named_parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    epochs = 1

    train_loss_list = []
    prog_bar = tqdm(train_loader, total=len(train_loader))
    print("\n\nBEGIN TRAINING\n")
    # TRAINING LOOP
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(prog_bar):
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
            print(f"Epoch: {epoch}, Loss: {losses}")

    print("FINISH TRAINING")
    # Save the model
    torch.save(model.state_dict(), "fasterrcnn_1epoch.pth")

if __name__ == "__main__":
    main()