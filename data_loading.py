import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
from os.path import abspath
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from monai.data import CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ResizeD, NormalizeIntensityd, MapTransform, RandSpatialCropd, RandFlipd, RandRotated, RandAffined
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import logical_or, stack


### TODO:
# Bas - implement loading of BRATS dataset
# Eva - Expand on transformations
# Someone - Implement a way to setup the dataset for 2D data - lars is picking this up
# We'll be trying to implement this: https://github.com/Project-MONAI/tutorials/blob/main/modules/2d_inference_3d_volume.ipynb
# Lars - make dataloader output matrixes 

# Shape of tensors in the batch is [batch_size, channels, Width, Depth, Height]


# Get the absolute path of the current file (data_loading.py)
CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR  # Adjust if data_loading.py is nested deeper

class MRIDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the WMH dataset.
    Handles data collection, transformations, and DataLoader creation.
    """

    def __init__(
        self,
        include_keys: list = ["flair", "t1", "WMH"],
        dataset: str = "WMH",
        data_dir_wmh: str = "data/WMH",
        data_dir_brats: str = "data/BraTS",
        batch_size: int = 8,
        num_workers: int = 8,
    ):
        """
        Initialize the data module to load either the wmh dataset or the BraTS dataset.

        Parameters
        ----------
        data_dir : str
            Path to the root directory of the WMH dataset.
        batch_size : int
            Batch size for DataLoaders.
        num_workers: int
            Number of subprocesses to use during data loading, decrease for slightly less memory usage.
        """
        super().__init__()
        
        if dataset.lower() == "wmh":
            # set data_dir
            data_dir = data_dir_wmh
            
            # Check if keys are correct
            if include_keys: # if include keys not empty 
                if any([key not in ["flair", "t1", "WMH"] for key in include_keys]):
                    raise MisconfigurationException(f'Unsupported key for WMH dataset, use any combination of ["flair", "t1", "WMH"]')
                else:
                    self.include_keys = include_keys
            
        elif dataset.lower() == "brats":
            # set data_dir
            data_dir = data_dir_brats

            # Check if keys are correct
            if include_keys: # if include keys not empty 
                if any([key not in ["t1ce", "t2", "flair", "t1", "seg"] for key in include_keys]):
                    raise MisconfigurationException(f'Unsupported key for brats dataset, use any combination of ["t1ce", "t2", "flair", "t1", "seg"]')
                else:
                    self.include_keys = include_keys
        else:
            raise MisconfigurationException(f'"{dataset}" is not a supported dataset, use either "WMH" or "brats"')

        self.data_dir = REPO_DIR.joinpath(data_dir).resolve()
        self.dataset = dataset.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transformations
        self.train_transforms_wmh_3D = Compose([
            LoadImaged(keys=["flair", "t1", "WMH"]),
            EnsureChannelFirstd(keys=["flair", "t1", "WMH"]),
            RandSpatialCropd(keys=["flair", "t1", "WMH"], roi_size=(32, 32, 32)),  # Crop all to the same shape
            RandFlipd(keys=["flair", "t1", "WMH"], prob=0.5, spatial_axis=0), 
            RandRotated(
                keys=["flair", "t1", "WMH"], 
                range_x=(-30, 30),  # Rotation range in degrees around x-axis
                range_y=(-30, 30),  # Rotation range in degrees around y-axis
                range_z=(-30, 30),  # Rotation range in degrees around z-axis
                prob=0.5,           # Probability of applying the rotation
                mode=("bilinear", "bilinear", "nearest"),  # Interpolation for flair, t1, and WMH
                align_corners=True
            ), 
            RandAffined(keys=["flair", "t1", "WMH"], prob=0.5, rotate_range=(0, 30), shear_range=(0.05, 0.1), scale_range=(0.9, 1.1)), 
            #mt.ResizeD(keys=["flair", "t1", "WMH"], spatial_size=(128, 128, 128)),  # Resize all to the same shape #TODO - remove?
            NormalizeIntensityd(keys=["flair", "t1"])
        ])


        self.val_transforms_wmh_3D = Compose([
            LoadImaged(keys=["flair", "t1", "WMH"]),
            EnsureChannelFirstd(keys=["flair", "t1", "WMH"]),
            ResizeD(keys=["flair", "t1", "WMH"], spatial_size=(128, 128, 128)), # NEEDS TO BE REMOVED WHEN A BETTER SOLUTION GETS IMPLEMENTED
            NormalizeIntensityd(keys=["flair", "t1"])
        ])

        self.train_transforms_brats = Compose([
            LoadImaged(keys=["t1ce", "t2", "flair", "t1", "seg"]),
            EnsureChannelFirstd(keys=["t1ce", "t2", "flair", "t1", "seg"]),
            ResizeD(keys=["flair", "t1", "seg"], spatial_size=(128, 128, 128)), # NEEDS TO BE REMOVED WHEN A BETTER SOLUTION GETS IMPLEMENTED
            NormalizeIntensityd(keys=["t1ce", "t2", "flair", "t1"])
        ])

    def collect_samples_wmh(self, root: Path) -> list[dict]:
        """
        Collect all the valid sample paths from the wmh dataset.

        Parameters
        ----------
        root : Path
            Root path of the dataset.

        Returns
        -------
        list[dict]
            List of dictionaries with paths to FLAIR, T1, and WMH files.
        """
        samples = []
        for city in ["Utrecht", "Singapore", "Amsterdam"]:
            city_dir = root.joinpath(city)
            for subject_dir in city_dir.iterdir():
                if subject_dir.is_dir():
                    flair_path = subject_dir / "FLAIR.nii.gz"
                    t1_path = subject_dir / "T1.nii.gz"
                    wmh_path = subject_dir / "wmh.nii.gz"

                # Ensure all necessary files exist
                if flair_path.exists() and t1_path.exists() and wmh_path.exists():                    
                    samples.append({"flair": flair_path, "t1": t1_path, "WMH": wmh_path})
                else:
                    print(f"Missing files in {subject_dir}")

        # remove file paths that are not in include_keys
        filtered_samples = []
        # print(f"include keys: {self.include_keys}") # DEBUG
        # print(f"dataset keys: {list(samples[0].keys())}") # DEBUG
        for sample in samples:
            for key in sample.keys():
                if key not in self.include_keys:
                    sample.pop(key)
                filtered_samples.append(sample)
                
                    
        return filtered_samples
    
    def collect_samples_brats(self, root: Path) -> list[dict]:

        brats_data_dir = root.joinpath("Data")
        samples = []

        for subject_dir in brats_data_dir.iterdir():
            if subject_dir.is_dir():
                file_paths = subject_dir.glob("*.nii.gz")

                # Assign file to name
                t1ce_path, flair_path, t2_path, seg_path, t1_path = (None, None, None, None, None)
                for file_path in file_paths:
                    if file_path.stem.split("_")[-1] == "t1ce.nii":
                        t1ce_path = file_path
                    elif file_path.stem.split("_")[-1] == "flair.nii":
                        flair_path = file_path
                    elif file_path.stem.split("_")[-1] == "t2.nii":
                        t2_path = file_path
                    elif file_path.stem.split("_")[-1] == "seg.nii":
                        seg_path = file_path
                    elif file_path.stem.split("_")[-1] == "t1.nii":
                        t1_path = file_path
                    else:
                        print(f"Unexpected file in {subject_dir}: {file_path}")

                # Ensure all necessary files exist
                if t1ce_path and flair_path and t2_path and seg_path and t1_path:
                    samples.append({
                        "t1ce": t1ce_path,
                        "flair": flair_path,
                        "t2": t2_path,
                        "seg": seg_path,
                        "t1": t1_path
                    })

        # remove file paths that are not in include_keys
        converted_samples = []
        for sample in samples:
            seg_path = sample.get("seg")
            if seg_path:
                seg_image = LoadImaged(keys=["seg"])({"seg": seg_path})["seg"]

                # Apply label combination logic based on new label mapping
                label_channels = [
                    seg_image == 1,  # Necrotic and non-enhancing tumor core
                    seg_image == 2,  # Peritumoral edema
                    seg_image == 4,  # Enhancing tumor
                    logical_or(logical_or(seg_image == 1, seg_image == 4), seg_image == 2)  # Whole tumor
                ]

                # Stack channels to create multi-channel segmentation
                multi_channel_label = stack(label_channels, dim=0).float()

                # Save the processed segmentation into the sample
                sample["multi_channel_seg"] = multi_channel_label

            # Filter the sample to include only necessary keys
            filtered_sample = {k: v for k, v in sample.items() if k in self.include_keys or k == "multi_channel_seg"}
            converted_samples.append(filtered_sample)

        return converted_samples

    def setup(self, stage: str = None):
        """
        Load and split the data into train, val, and test sets.

        Parameters
        ----------
        stage : str
            Stage of training: 'fit', 'test', etc.
        """
        # Collect sample paths from the the chosen dataset
        all_samples = []
        if self.dataset.lower() == "wmh":
            all_samples.extend(self.collect_samples_wmh(self.data_dir))
        elif self.dataset.lower() == "brats":
            all_samples.extend(self.collect_samples_brats(self.data_dir))
        print("### all_samples:", all_samples)
        
        # Split into train, validation, and test sets
        if self.dataset.lower() == "wmh":
            # print(f"Amount of samples in dataset: {len(all_samples)}") # DEBUG
            train_samples, temp_samples = train_test_split(all_samples, train_size=0.8, random_state=42, shuffle=True)
            val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42, shuffle=True)

            # Cache the datasets for performance
            self.train_dataset = CacheDataset(train_samples, transform=self.train_transforms_wmh_3D)
            self.val_dataset = CacheDataset(val_samples, transform=self.val_transforms_wmh_3D)
            self.test_dataset = CacheDataset(test_samples, transform=self.val_transforms_wmh_3D)  # Same transforms for test

        elif self.dataset.lower() == "brats": # brats is used for pre_training so no split is required
            train_samples, val_samples = train_test_split(all_samples, train_size=0.8, random_state=42, shuffle=True)

            # Cache the dataset for performance
            self.train_dataset = CacheDataset(train_samples, transform=self.train_transforms_brats)
            self.val_dataset = CacheDataset(val_samples, transform=self.train_transforms_brats)
            
            
    def train_dataloader(self):
        """
        Training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        """
        Validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        """
        Test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)