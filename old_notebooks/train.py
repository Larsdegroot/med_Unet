from pytorch_lightning import Trainer
from data_loading import WMHDataModule
from pathlib import Path
from model import *

# Path to dataset
data_dir = Path("data/WMH")

# Initialize the DataModule
datamodule = WMHDataModule(data_dir=data_dir, batch_size=4)

# Instantiate the LightningModule
model = LitUNet(
    n_dims=3,
    in_channels=1,
    out_channels=1,
    base_channels=8,
    depth=3,
    use_transpose=False,
    use_normalization=True,
    learning_rate=1e-3,
)

# Train with PyTorch Lightning
trainer = Trainer(
    max_epochs=1, 
    accelerator="auto",
    log_every_n_steps=2
)
