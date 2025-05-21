# utils/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index, path

def get_dataloaders(data_dir, batch_size=16):
    # 강한 증강 transform (train/val/test 동일하게 적용)
    shared_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomApply([
            transforms.GaussianBlur(3)
        ], p=0.2),
        transforms.RandomApply([
            transforms.RandomPosterize(bits=4)
        ], p=0.2),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'valid')
    test_path = os.path.join(data_dir, 'test')

    train_dataset = IndexedImageFolder(train_path, transform=shared_transform)
    val_dataset = IndexedImageFolder(val_path, transform=shared_transform)
    test_dataset = IndexedImageFolder(test_path, transform=shared_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes