from data_loading import WMHDataModule
from model import LitUNet
from lightning.pytorch.cli import LightningCLI

def cli_main():
    cli = LightningCLI(LitUNet, WMHDataModule)


if __name__ == "__main__":
    cli_main()