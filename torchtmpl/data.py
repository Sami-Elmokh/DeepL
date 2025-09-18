# coding: utf-8

# imports

import logging
from pathlib import Path

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
from torchvision import transforms

# from an external git repository >> to be added as requirement
# or to  include whole thing here

from . import data_loading


# from GLC._old.data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset    
import albumentations as A
from albumentations.pytorch import ToTensorV2




def get_train_transforms():
    return A.Compose([
            # RandomResizedCrop(256, 256),
            # Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # CoarseDropout(p=0.5),
            # Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)


def get_valid_transforms():
    return A.Compose([
            A.CenterCrop(256, 256, p=1.),
            # Resize(CFG['img_size'], CFG['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)





class GeoLifeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        subset,
        *,
        region="both",
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        if subset == "test":
            subset_file_suffix = "test"
            self.training_data = False
        else:
            subset_file_suffix = "train"
            self.training_data = True

        df_fr = pd.read_csv(
            self.root
            / "observations"
            / "observations_fr_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id", nrows=50000
        )
        df_us = pd.read_csv(
            self.root
            / "observations"
            / "observations_us_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id", nrows=50000
        )

        if region == "both":
            df = pd.concat((df_fr, df_us))
        elif region == "fr":
            df = df_fr
        elif region == "us":
            df = df_us

        if self.training_data and subset != "train+val":
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values

        if self.training_data:
            self.targets = df.species_id.values 
        else:
            self.targets = None

        if use_rasters:
            if patch_extractor is None:

                patch_extractor = data_loading.PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def __len__(self):
        return len(self.observation_ids)

    # Adapted __getitem__ method
    def __getitem__(self, index):
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        try:
            patches = data_loading.load_patch(observation_id, self.root, data=self.patch_data)
        except ValueError:
            pass
        #normalize patches.copy()[0] and 1 or 2 each on their own
        patches_copy= patches.copy()
        normalized_image_list = []

        for image in patches_copy:
            # Calculate min and max values for each image
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val - min_val == 0:
                val_norm = 1
            else:
                val_norm = max_val - min_val
            # Apply Min-Max normalization
            normalized_image = (image - min_val) / val_norm
        
            # Append normalized image to the list
            normalized_image_list.append(normalized_image)        
        patch_1 = [torch.from_numpy(normalized_image_list[0][:,:,i][None,:,:]) for i in range(3)]
        #print("patch_1",torch.max(patch_1[0]))
        patch_2 = [torch.from_numpy(arr.copy()[None,:,:]) for arr in normalized_image_list[1:]]
        #print("patch_2",torch.max(patch_2[0]))

        patches = torch.cat(patch_1 + patch_2, dim=0)  # Concatenate the patches along the channel dimension


        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)].copy()
            for i in range(environmental_patches.shape[0]):  # Iterate over channels
                channel_data = environmental_patches[i]
                if np.isnan(channel_data).any():  # Check if NaN values are present
                    # Compute the mean excluding NaN values
                    channel_mean = np.nanmean(channel_data)
                    # Replace NaN values with the computed mean
                    channel_data[np.isnan(channel_data)] = channel_mean
                else:
                    # If there are no NaN values, directly compute the mean
                    channel_mean = np.mean(channel_data)
                environmental_patches[i][np.isnan(environmental_patches[i])] = channel_mean

                channel_min = np.min(environmental_patches[i])
                channel_max = np.max(environmental_patches[i])
                if (channel_max - channel_min == 0):
                    normalize=1
                else:
                    normalize = channel_max - channel_min
                environmental_patches[i] = (environmental_patches[i] - channel_min) / normalize
            environmental_patches_tensor = torch.Tensor(environmental_patches)
            #patches = patches + torch.Tensor(environmental_patches)
            #print("env patch",torch.max(environmental_patches_tensor))
            
            patches = torch.cat((patches,environmental_patches_tensor), dim=0)
            #print("final patch",torch.max(patches))



        if len(patches) == 1:
            patches = patches[0]

        # if self.transform:
        #     patches = self.transform(patches)

        if self.training_data:
            
            #target = torch.nn.functional.one_hot(torch.tensor(self.targets[index]), num_classes=17036)
            target = torch.tensor(self.targets[index])
            # if self.target_transform:
            #     target = self.target_transform(target)
            return patches, target
        else:
            return patches


def get_train_transforms():
    return A.Compose([
            # RandomResizedCrop(256, 256),
            # Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # CoarseDropout(p=0.5),
            # Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)


def get_valid_transforms():
    return A.Compose([
            A.CenterCrop(256, 256, p=1.),
            # Resize(CFG['img_size'], CFG['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)





class GeoLifeDataset_simple(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        subset,
        *,
        region="both",
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        if subset == "test":
            subset_file_suffix = "test"
            self.training_data = False
        else:
            subset_file_suffix = "train"
            self.training_data = True

        df_fr = pd.read_csv(
            self.root
            / "observations"
            / "observations_fr_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id", nrows=50000
        )
        df_us = pd.read_csv(
            self.root
            / "observations"
            / "observations_us_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id", nrows=50000
        )

        if region == "both":
            df = pd.concat((df_fr, df_us))
        elif region == "fr":
            df = df_fr
        elif region == "us":
            df = df_us

        if self.training_data and subset != "train+val":
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values

        if self.training_data:
            self.targets = df.species_id.values  # torch.tensor(df["species_id"].values, dtype = torch.long)
        else:
            self.targets = None

        # FIXME: add back landcover one hot encoding?
        # self.one_hot_size = 34
        # self.one_hot = np.eye(self.one_hot_size)

        if use_rasters:
            if patch_extractor is None:

                patch_extractor = data_loading.PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def __len__(self):
        return len(self.observation_ids)

    # Adapted __getitem__ method
    def __getitem__(self, index):
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        try:
            patches = data_loading.load_patch(observation_id, self.root, data=self.patch_data)
        except ValueError:
            pass
        #normalize patches.copy()[0] and 1 or 2 each on their own
        patches_copy= patches.copy()
        normalized_image_list = []

        for image in patches_copy[1:3]:
            # Calculate min and max values for each image
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val - min_val == 0:
                val_norm = 1
            else:
                val_norm = max_val - min_val
            # Apply Min-Max normalization
            normalized_image = (image - min_val) / val_norm
        
            # Append normalized image to the list
            normalized_image_list.append(normalized_image)        

        patch_2 = [torch.from_numpy(arr.copy()[None,:,:]) for arr in normalized_image_list]


        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)].copy()
            for i in range(1):  # Iterate over channels
                channel_data = environmental_patches[i]
                if np.isnan(channel_data).any():  # Check if NaN values are present
                    # Compute the mean excluding NaN values
                    channel_mean = np.nanmean(channel_data)
                    # Replace NaN values with the computed mean
                    channel_data[np.isnan(channel_data)] = channel_mean
                else:
                    # If there are no NaN values, directly compute the mean
                    channel_mean = np.mean(channel_data)
                environmental_patches[i][np.isnan(environmental_patches[i])] = channel_mean

                channel_min = np.min(environmental_patches[i])
                channel_max = np.max(environmental_patches[i])
                if (channel_max - channel_min == 0):
                    normalize=1
                else:
                    normalize = channel_max - channel_min
                environmental_patches[i] = (environmental_patches[i] - channel_min) / normalize
            environmental_patches_tensor = torch.Tensor(environmental_patches[0][None,:,:])

            patches = torch.cat((patch_2[0].clone().detach(), patch_2[1].clone().detach(),environmental_patches_tensor), dim=0)



        if len(patches) == 1:
            patches = patches[0]

        if self.transform:
            patches = self.transform(patches)

        if self.training_data:
            
            #target = torch.nn.functional.one_hot(torch.tensor(self.targets[index]), num_classes=17036)
            target = torch.tensor(self.targets[index])

            if self.target_transform:
                target = self.target_transform(target)
            return patches, target
        else:
            return patches, observation_id 


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


def get_dataloaders(sample, data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    # input_transform = transforms.Compose(
    #     [transforms.Grayscale(), transforms.Resize((128, 128)),
    #      transforms.ToTensor()]
    # )

    # base_dataset = torchvision.datasets.Caltech101(
    #     root=data_config["trainpath"],
    #     download=True,
    #     transform=input_transform,
    # )
    if not sample:
        base_dataset = GeoLifeDataset(root=data_config["trainpath"],
                                    subset="train+val",
                                    region="both",
                                    patch_data="all",
                                    use_rasters=True,
                                    patch_extractor=None,
                                    transform=False,
                                    target_transform=False)  # add transformations
    else:
        base_dataset = GeoLifeDataset_simple(root=data_config["trainpath"],
                                    subset="train+val",
                                    region="both",
                                    patch_data="all",
                                    use_rasters=True,
                                    patch_extractor=None,
                                    transform=False,
                                    target_transform=False)  # add transformations

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    return train_loader, valid_loader



def get_test_dataloaders(data_config, use_cuda,sample=True):

    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    # input_transform = transforms.Compose(
    #     [transforms.Grayscale(), transforms.Resize((128, 128)),
    #      transforms.ToTensor()]
    # )

    # base_dataset = torchvision.datasets.Caltech101(
    #     root=data_config["trainpath"],
    #     download=True,
    #     transform=input_transform,
    # )
    if sample:
        test_dataset = GeoLifeDataset_simple(root=data_config["trainpath"],
                                    subset="test",
                                    region="both",
                                    patch_data="all",
                                    use_rasters=True,
                                    patch_extractor=None,
                                    transform=False,
                                    target_transform=False)  # add transformations
    else:
        test_dataset = GeoLifeDataset(root=data_config["trainpath"],
                                    subset="test",
                                    region="both",
                                    patch_data="all",
                                    use_rasters=True,
                                    patch_extractor=None,
                                    transform=False,
                                    target_transform=False)  # add transformations

    logging.info(f"  - I loaded {len(test_dataset)} samples")

    #indices = list(range(len(test_dataset)))
    #test_dataset = torch.utils.data.Subset(base_dataset, indices)

    # Build the dataloaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )


    return test_loader

