# med_Unet
This is a implementation of a Unet Imagine network to segment brain lessions in MRI images of the brain. This is a project done by students for the course AI for medical imaging.

## Environment setup

Recreate the python virtual environment by running:
```
pip install -r requirements.txt
```
This code is reliant on [Pytorch](https://pytorch.org/) which can have different installation step depending on hardware. Please follow your installation requirements by following the step on the [pytorch installation page.](https://pytorch.org/get-started/locally/)

## Use
Our code makes use of [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) and it's [command line interface](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) (CLI). 

See help for the CLI:
```
python main.py --help
```

To print the config and see which parameters can be changed:
```
python main.py --print_config
```

Make a default config:
```
python main.py --print_config > path/to/config.yaml
```

To Train a model:
```
python main.py fit --config path/to/config.yaml
```

To run validation:
```
python main.py validate --config path/to/config.yaml
```

To run testing:
```
python main.py test --config path/to/config.yaml
```

## Code structure
```
.
├── README.md
├── configs # Configs for the experiments
├── data # Where to store dataset, you need to create this yourself
├── data_loading.py # Code to load the dataset
├── main.py # Code for the CLI that runs either training, validation or testing
├── old_notebooks # Old notebooks left from during development
├── requirements.txt # Environment requirements
└── run_experiment_colab.ipynb # Google Colab notebook to run a experiment
```
