#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 06:53:45 2025

@author: adolfo
"""
import datetime

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ResNet101 import model, train_function, validate_function, criterion, optimizer 


device = torch.device("cpu")

num_classes = 18
batch_size = 32
learning_rate = 0.001
num_epochs = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dir = '/home/adolfo/Documents/TFG/DATASETS/Species - Mediterráneo/Training_Set'
val_dir = '/home/adolfo/Documents/TFG/DATASETS/Species - Mediterráneo/Test_Set'

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\nTraining stated at: {datetime.datetime.now().strftime('%H:%M')}\n" )

for epoch in range(num_epochs):
    train_loss, train_acc = train_function(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_function(model, val_loader, criterion)

    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


torch.save(model.state_dict(), "resnet101_fish_classifier.pth")

print(f"\nTraining ended at: {datetime.datetime.now().strftime('%H:%M')}\n" )
