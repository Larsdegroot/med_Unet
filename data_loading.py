import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from pathlib import Path
from os.path import abspath
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from monai.data import CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ResizeD, NormalizeIntensityd
from lightning.pytorch.utilities.exceptions import MisconfigurationException


### TODO:
# Bas - implement loading of BRATS dataset
# Eva - Expand on transformations
# Someone - Implement a way to setup the dataset for 2D data

class MRIDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the WMH dataset.
    Handles data collection, transformations, and DataLoader creation.
    """

    def __init__(
        self,
        dataset: str = "WMH",
        data_dir_wmh: str = "data/wmh",
        data_dir_brats: str = "data/BraTS"
        batch_size: int = 8,
        num_workers: int = 8
    ):
        """
        Initialize the data module.

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
            data_dir = data_dir_wmh
        elif dataset.lower() == "brats":
            data_dir = data_dir_brats
        else:
            raise MisconfigurationException(f'"{dataset}" is not a supported dataset, use either "WMH" or "brats"')
            
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transformations
        self.train_transforms = Compose([
            LoadImaged(keys=["flair", "t1", "WMH"]),
            EnsureChannelFirstd(keys=["flair", "t1", "WMH"]),
            ResizeD(keys=["flair", "t1", "WMH"], spatial_size=(128, 128, 128)), # NEEDS TO BE REMOVED WHEN A BETTER SOLUTION GETS IMPLEMENTED
            NormalizeIntensityd(keys=["flair", "t1"])
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=["flair", "t1", "WMH"]),
            EnsureChannelFirstd(keys=["flair", "t1", "WMH"]),
            ResizeD(keys=["flair", "t1", "WMH"], spatial_size=(128, 128, 128)), # NEEDS TO BE REMOVED WHEN A BETTER SOLUTION GETS IMPLEMENTED
            NormalizeIntensityd(keys=["flair", "t1"])
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
        for subject_dir in root.iterdir():
            if subject_dir.is_dir():
                flair_path = subject_dir / "FLAIR.nii.gz"
                t1_path = subject_dir / "T1.nii.gz"
                wmh_path = subject_dir / "wmh.nii.gz"

                # Ensure all necessary files exist
                if flair_path.exists() and t1_path.exists() and wmh_path.exists():
                    samples.append({"flair": flair_path, "t1": t1_path, "WMH": wmh_path})
                else:
                    print(f"Missing files in {subject_dir}")
        return samples

    def collect_samples_brats(self, root: Path) -> list[dict]:
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
        for subject_dir in root.iterdir():
            if subject_dir.is_dir():
                flair_path = subject_dir / "FLAIR.nii.gz"
                t1_path = subject_dir / "T1.nii.gz"
                wmh_path = subject_dir / "wmh.nii.gz"

                # Ensure all necessary files exist
                if flair_path.exists() and t1_path.exists() and wmh_path.exists():
                    samples.append({"flair": flair_path, "t1": t1_path, "WMH": wmh_path})
                else:
                    print(f"Missing files in {subject_dir}")
        return samples

    def setup(self, stage: str = None):
        """
        Load and split the data into train, val, and test sets.

        Parameters
        ----------
        stage : str
            Stage of training: 'fit', 'test', etc.
        """
        # Collect sample paths from the dataset
        all_samples = []
        for subject_path in self.data_dir.iterdir():
            if subject_path.is_dir():
                all_samples.extend(self.collect_samples_wmh(subject_path))

        # Split into train, validation, and test sets
        if self.
        train_samples, temp_samples = train_test_split(all_samples, train_size=0.8, random_state=42, shuffle=True)
        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42, shuffle=True)

        # Cache the datasets for performance
        self.train_dataset = CacheDataset(train_samples, transform=self.train_transforms)
        self.val_dataset = CacheDataset(val_samples, transform=self.val_transforms)
        self.test_dataset = CacheDataset(test_samples, transform=self.val_transforms)  # Same transforms for test

    def train_dataloader(self):
        """
        Training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)