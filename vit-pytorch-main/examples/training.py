import torch
print(torch.__version__)

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from vision_transformer_pytorch import VisionTransformer
print(f"Torch: {torch.__version__}")

# Training settings
batch_sz = 64
num_epochs = 6
learning_rate = 3e-5
lr_decay_factor = 0.7
random_seed = 42
device_type = 'cuda'
train_folder = 'data/train'
test_folder = 'data/test'
saved_model_path = "vit_model"

class AnimalDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img_path = self.file_paths[index]
        image = Image.open(img_path)
        transformed_image = self.transform(image)

        category = img_path.split("/")[-1].split(".")[0]
        category_label = 1 if category == "dog" else 0

        return transformed_image, category_label

train_data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

validation_data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

test_data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

def set_seed(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

def save_model_to_disk(model, file_path):
    """
    Save the PyTorch model to the specified path.

    Parameters:
        model (torch.nn.Module): The model to save.
        file_path (str): The file path to save the model.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model_from_disk(model, file_path):
    """
    Load a PyTorch model's state_dict from the specified path.

    Parameters:
        model (torch.nn.Module): The model instance to load the state_dict into.
        file_path (str): The file path to load the model from.

    Returns:
        torch.nn.Module: The model with the loaded state_dict.
    """
    model.load_state_dict(torch.load(file_path))
    print(f"Model loaded from {file_path}")
    return model

set_seed(random_seed)
os.makedirs('data', exist_ok=True)

with zipfile.ZipFile('train.zip') as train_zip_file:
    train_zip_file.extractall('data')
    
with zipfile.ZipFile('test.zip') as test_zip_file:
    test_zip_file.extractall('data')

# Data processing
train_image_list = glob.glob(os.path.join(train_folder, '*.jpg'))
test_image_list = glob.glob(os.path.join(test_folder, '*.jpg'))
print(f"Train Data: {len(train_image_list)}")
print(f"Test Data: {len(test_image_list)}")

train_labels = [path.split('/')[-1].split('.')[0] for path in train_image_list]
train_image_list, validation_image_list = train_test_split(
    train_image_list, 
    test_size=0.2,
    stratify=train_labels,
    random_state=random_seed
)

print(f"Train Data: {len(train_image_list)}")
print(f"Validation Data: {len(validation_image_list)}")
print(f"Test Data: {len(test_image_list)}")

# Load datasets
training_dataset = AnimalDataset(train_image_list, transform=train_data_transforms)
validation_dataset = AnimalDataset(validation_image_list, transform=test_data_transforms)
testing_dataset = AnimalDataset(test_image_list, transform=test_data_transforms)

training_loader = DataLoader(dataset=training_dataset, batch_size=batch_sz, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_sz, shuffle=True)
testing_loader = DataLoader(dataset=testing_dataset, batch_size=batch_sz, shuffle=True)

print(len(training_dataset), len(training_loader))
print(len(validation_dataset), len(validation_loader))

# Efficient Attention
efficient_attention = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

# Visual Transformer
vision_transformer = VisionTransformer(
    image_size=224, 
    patch_size=32, 
    num_classes=2, 
    dim=128, 
    depth=12, 
    mlp_dim=256, 
    num_heads=8, 
    channels=3, 
    dropout_rate=0.1, 
    attention_dropout_rate=0.1
).to(device_type)

# Training
# Loss function
loss_function = nn.CrossEntropyLoss()
# Optimizer
adam_optimizer = optim.Adam(vision_transformer.parameters(), lr=learning_rate)
# Scheduler
learning_rate_scheduler = StepLR(adam_optimizer, step_size=1, gamma=lr_decay_factor)

for epoch_idx in range(num_epochs):
    training_loss = 0
    training_accuracy = 0

    for batch_data, batch_labels in tqdm(training_loader):
        batch_data = batch_data.to(device_type)
        batch_labels = batch_labels.to(device_type)

        predictions = vision_transformer(batch_data)
        loss = loss_function(predictions, batch_labels)

        adam_optimizer.zero_grad()
        loss.backward()
        adam_optimizer.step()

        batch_accuracy = (predictions.argmax(dim=1) == batch_labels).float().mean()
        training_accuracy += batch_accuracy / len(training_loader)
        training_loss += loss / len(training_loader)

    with torch.no_grad():
        validation_accuracy = 0
        validation_loss = 0
        for batch_data, batch_labels in validation_loader:
            batch_data = batch_data.to(device_type)
            batch_labels = batch_labels.to(device_type)

            val_predictions = vision_transformer(batch_data)
            val_loss = loss_function(val_predictions, batch_labels)

            batch_accuracy = (val_predictions.argmax(dim=1) == batch_labels).float().mean()
            validation_accuracy += batch_accuracy / len(validation_loader)
            validation_loss += val_loss / len(validation_loader)

    print(
        f"Epoch : {epoch_idx + 1} - loss : {training_loss:.4f} - acc: {training_accuracy:.4f} - val_loss : {validation_loss:.4f} - val_acc: {validation_accuracy:.4f}\n"
    )

    save_model_to_disk(vision_transformer, file_path=saved_model_path)
