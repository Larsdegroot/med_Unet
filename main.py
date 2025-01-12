from data_loading import MRIDataModule
from model import LitUNet
from lightning.pytorch.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
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
    cli = LightningCLI(LitUNet, MRIDataModule, save_config_overwrite=True)

    # config logger using config file
    log_config = cli.config['trainer']['logger']
    logger = TensorBoardLogger(save_dir=log_config['save_dir'], name=log_config['name'])
    cli.trainer.logger = logger

    # config data folders using config file
    data_path_wmh = cli.config['data']['data_dir_wmh']
    cli.datamodule.data_dir_wmh = data_path_wmh
    data_path_brats = cli.config['data']['data_dir_brats']
    cli.datamodule.data_dir_brats = data_path_brats

if __name__ == "__main__":
    cli_main()
