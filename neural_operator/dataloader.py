import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    """Lazy-loading dataset for .npy gravitational lensing images."""

    def __init__(self, folder_path, transforms=None, max_samples=None):
        self.folder_path = folder_path
        self.transform = transforms
        self.class_folders = sorted(
            [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        )

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_folders)

        # Store (file_path, class_name) pairs — lazy loading
        self.samples = []
        for class_folder in self.class_folders:
            class_path = os.path.join(folder_path, class_folder)
            files = sorted([f for f in os.listdir(class_path) if f.endswith('.npy')])
            if max_samples is not None:
                files = files[:max_samples]
            for file_name in files:
                self.samples.append((os.path.join(class_path, file_name), class_folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_name = self.samples[idx]
        data_point = np.load(file_path)
        label = self.label_encoder.transform([class_name])[0]

        # Z-score normalization per channel
        mean = np.mean(data_point, axis=(1, 2), keepdims=True)
        std = np.std(data_point, axis=(1, 2), keepdims=True)
        data_point = (data_point - mean) / (std + 1e-8)

        data_point = torch.from_numpy(data_point).float()

        if self.transform:
            data_point = self.transform(data_point)

        return data_point, torch.tensor(label, dtype=torch.long)


def get_dataloaders(train_dir, image_size, batch_size, val_dir=None,
                    train_split=0.9, num_workers=2, seed=42, max_samples=None):
    """Create train/test/val DataLoaders with appropriate transforms.

    Args:
        train_dir: Path to training data (class subfolders with .npy files).
        image_size: Target image size for resizing.
        batch_size: Batch size for DataLoaders.
        val_dir: Optional separate validation directory. If None, no val loader.
        train_split: Fraction of train_dir used for training (rest is test).
        num_workers: Number of DataLoader workers.
        seed: Random seed for reproducible splits.
        max_samples: Optional max samples per class.

    Returns:
        dict with 'train', 'test', and optionally 'val' DataLoaders,
        plus 'class_names' list.
    """
    train_augmentation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
    ])

    eval_augmentation = transforms.Compose([transforms.Resize(image_size)])

    # Train + test split
    train_dataset_aug = CustomDataset(train_dir, transforms=train_augmentation, max_samples=max_samples)
    train_size = int(train_split * len(train_dataset_aug))
    test_size = len(train_dataset_aug) - train_size
    train_dataset, _ = random_split(train_dataset_aug, [train_size, test_size],
                                    generator=torch.Generator().manual_seed(seed))

    # Test split with eval transforms (no augmentation)
    test_dataset_eval = CustomDataset(train_dir, transforms=eval_augmentation, max_samples=max_samples)
    _, test_dataset = random_split(test_dataset_eval, [train_size, test_size],
                                   generator=torch.Generator().manual_seed(seed))

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }

    # Validation set (separate folder)
    if val_dir and os.path.isdir(val_dir):
        val_dataset = CustomDataset(val_dir, transforms=eval_augmentation, max_samples=max_samples)
        loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    # Also provide a train-eval loader (no augmentation, for clean evaluation)
    train_eval_dataset = CustomDataset(train_dir, transforms=eval_augmentation, max_samples=max_samples)
    train_eval_subset, _ = random_split(train_eval_dataset, [train_size, test_size],
                                        generator=torch.Generator().manual_seed(seed))
    loaders['train_eval'] = DataLoader(train_eval_subset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

    class_names = list(train_dataset_aug.label_encoder.classes_)
    return loaders, class_names
