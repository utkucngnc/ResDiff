import os
import cv2
import numpy as np
from typing import Any
from datetime import datetime
from tqdm import tqdm

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from loss import CNNLoss
import backbone as model
import config
import imgproc

def build_model() -> nn.Module:
    simplesr_model = model.__dict__[config.model_arch_name](in_ch=config.in_channels,
                                                         out_ch=config.out_channels,
                                                         channels=config.channels)
    simplesr_model = simplesr_model.to(device=config.device)

    return simplesr_model

class CustomTrainDataset(Dataset):
    def __init__(
            self,
            gt_image_dir: str,
            gt_image_size: int,
            upscale_factor: int,
            mode: str
    ) -> None:
        super(CustomTrainDataset, self).__init__()
        self.image_file_names = [os.path.join(gt_image_dir, image_file_name) for image_file_name in
                                 os.listdir(gt_image_dir)]
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode
    
    def __getitem__(self, index) -> Any:
        # Read a batch of image data
        gt_crop_image = cv2.imread(self.image_file_names[index]).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "Train":
            gt_crop_image = imgproc.random_crop(gt_crop_image, self.gt_image_size)
        elif self.mode == "Valid":
            gt_crop_image = imgproc.center_crop(gt_crop_image, self.gt_image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)

        # BGR convert Y channel
        gt_crop_y_image = imgproc.bgr_to_ycbcr(gt_crop_image, only_use_y_channel=True)
        lr_crop_y_image = imgproc.bgr_to_ycbcr(lr_crop_image, only_use_y_channel=True)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_crop_y_tensor = imgproc.image_to_tensor(gt_crop_y_image, False, False)
        lr_crop_y_tensor = imgproc.image_to_tensor(lr_crop_y_image, False, False)

        return {"gt": gt_crop_y_tensor, "lr": lr_crop_y_tensor}
    
    def __len__(self) -> int:
        return len(self.image_file_names)


def one_epoch_train(epoch: int,
                    tb_writer: SummaryWriter,
                    model: nn.Module, 
                    train_loader: DataLoader, 
                    criterion: CNNLoss, 
                    optimizer: th.optim.Optimizer,
                    device: th.device) -> float:
    last_loss = 0.0
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for _, data in enumerate(pbar):
        i, batch = data
        gt = batch["gt"].to(device)
        lr = batch["lr"].to(device)
        optimizer.zero_grad()
        preds = model(lr)
        loss = criterion(preds, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
    return last_loss


def main():
    # Create model
    simplesr_model = build_model().train()
    # Create loss function
    criterion = CNNLoss()
    # Create optimizer
    optimizer = th.optim.Adam(simplesr_model.parameters(), lr=config.model_lr)
    # Create dataset
    train_dataset = CustomTrainDataset(config.train_gt_images_dir, config.gt_image_size, config.upscale_factor, 'Train')
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    # Train model
    for epoch in range(config.epochs):
        loss = one_epoch_train(epoch, writer, simplesr_model, train_loader, criterion, optimizer, config.device)
        print(f"Epoch: {epoch}, Loss: {loss}")

if __name__ == "__main__":
    main()