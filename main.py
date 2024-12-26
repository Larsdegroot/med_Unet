from data_loading import MRIDataModule
from model import LitUNet
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.exceptions import MisconfigurationException

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.add_argument("--dataset", default="WMH")

#     def before_instantiate_classes(self):
#         if self.config["dataset"] == "WMH":
#             self.datamodule_class = WMHDataModule
#         elif self.config["dataset"].lower() == "brats":
#             self.datamodule_class = BratsDataModule
#         else:
#             raise MisconfigurationException(f'"{self.config["dataset"]}" is not a supported dataset, use either "WMH" or "brats" ')

def cli_main():
    cli = MyLightningCLI(LitUNet, MRIDataModule)


if __name__ == "__main__":
    cli_main()