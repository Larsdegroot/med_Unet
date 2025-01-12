import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from lightning import LightningModule
from typing import Callable, List
from monai.inferers import SlidingWindowInferer, SliceInferer, SimpleInferer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, recall_score

class DoubleConv(nn.Module):
    def __init__(
        self,
        n_dims: int,
        in_channels: int,
        out_channels: int,
        use_normalization: bool = True,
    ) -> None:
        super().__init__()

        # For now we are just considering 2d input
        # Thus expected input dimensions is [batch_size, channels, height, width] 
        # Or when not using batches [channels, height, width]
        # Or in the convention of pytorch: (N, C, H, W) or (C, H, W)

        # nn.Identity just return it's input so it's used as a replacement for normalization if normalization is not used
        # TO DO: find out what batchnorm does exactly
        # TO DO: Find out how exactly relu works
        
        conv = None
        norm = None

        if n_dims == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d if use_normalization else nn.Identity
        elif n_dims == 3:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d if use_normalization else nn.Identity
        else:
            raise ValueError("Invalid number of dimensions, Either choose 2 for 2 dimensional input or 3 for 3 dimensional input.")
        
        layers = [
            conv(in_channels , out_channels , kernel_size=3, padding=1),   
            norm(out_channels),                               
            nn.ReLU(inplace=True),                        
            conv(out_channels , out_channels , kernel_size=3, padding=1),         
            norm(out_channels),                                      
            nn.ReLU(inplace=True),                          
        ]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)
    
class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_dims: int,
        in_channels: int,
        out_channels: int,
        use_normalization: bool = True,
    ) -> None:
        super().__init__()

        pool = None
        if n_dims == 2:
            pool = nn.MaxPool2d
        elif n_dims == 3:
            pool = nn.MaxPool3d
        else:
            raise ValueError("Invalid number of dimensions, Either choose 2 for 2 dimensional input or 3 for 3 dimensional input.")
        
        self.encode = nn.Sequential(
            pool(kernel_size=2, stride=2),
            DoubleConv(n_dims, in_channels, out_channels, use_normalization),
        )

    def forward(self, x):
        return self.encode(x)

class DecoderBlock(nn.Module):
    '''
    Creates an instance of the decoder block in a UNet. The decoder block consist of the following:
    Upsample ->(+Skip connection) Convolutional layer -> Convolutional layer
    '''
    def __init__(
        self,
        n_dims: int,
        in_channels: int,
        out_channels: int,
        use_transpose: bool = False,
        use_normalization: bool = True,
    ) -> None:
        super().__init__()

        ### Pick upsample method
        conv = None

        ## 2 Dimensions
        if n_dims == 2:
            conv = nn.Conv2d

            # Two methods to upsample: transpose or with interpolation
            # Upsample the spatial dimensions (height, width) and reduce the number of channels by half.
            # Kernel size 2 and stride 2 for conv transpose is essential here to double the spatial dimensions
            if use_transpose:
                self.upsample = nn.ConvTranspose2d(in_channels , out_channels , kernel_size=2, stride=2)  # We cut the number of in-channels in half.
            else:
                # This upsample method uses a algorithm instead learned weights and can have artifects
                # The convolutional layer right after it is meant to deal with those artifacts
                self.upsample = nn.Sequential(  
                    nn.Upsample(scale_factor= 2, mode="bilinear", align_corners=True),
                    conv(in_channels , out_channels , kernel_size=1, padding=0),  
                )

        ## 3 Dimensions
        elif n_dims == 3:
            conv = nn.Conv3d

            # Same as with the 2D case. 
            if use_transpose:
                self.upsample = nn.ConvTranspose3d(in_channels, out_channels , kernel_size=2, stride=2)
            else:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor= 2, mode="trilinear", align_corners=True),
                    conv(in_channels, out_channels, kernel_size=1, padding=0),
                )

        ## Just
        self.decode = DoubleConv(n_dims, in_channels, out_channels, use_normalization)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # This is the skip connection
        x = torch.cat((x, skip), dim=1)
        x = self.decode(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        n_dims: int,
        in_channels: int,
        out_channels: int,
        base_channels: int = 8,
        depth: int = 4,
        use_transpose: bool = False,
        use_normalization: bool = True,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.n_dims = n_dims
        self.depth = depth

        if depth < 2:
            raise ValueError("Model depth must be 2 or greater")
        
        # Define the input layer
        layers = [DoubleConv(n_dims, in_channels , base_channels, use_normalization)]
        
        
        # Define the encoder path: it progressively doubles the number of channels
        current_features = base_channels
        for _ in range(depth - 1):
            layers.append(EncoderBlock(n_dims, current_features, current_features*2, use_normalization))
            current_features *= 2

        # Define the decoder path: progressively halves the number of channels
        for _ in range(depth - 1):
            layers.append(DecoderBlock(n_dims, current_features, current_features//2, use_transpose, use_normalization))
            current_features //= 2
        
        # Define the output layer
        if n_dims == 2:
            layers.append(nn.Conv2d(current_features, out_channels, kernel_size=1))
        elif n_dims == 3:
            layers.append(nn.Conv3d(current_features, out_channels, kernel_size=1))
        else:
            raise ValueError("Invalid number of dimensions")
        
        self.layers = nn.ModuleList(layers)
        if final_activation is not None:
            self.final_activation = final_activation
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = [self.layers[0](x)]

        # Encoder path
        # Pretty simple, just loop over the encoder blocks
        for layer in self.layers[1:self.depth]:
            xi.append(layer(xi[-1]))

        
        # Decoder path
        # We need to loop over the decoder blocks, but also need to
        # keep track of the skip connections
        for i, layer in enumerate(self.layers[self.depth:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        
        return self.final_activation(self.layers[-1](xi[-1]))

class LitUNet(LightningModule):
    '''
    Lighting version of the Unet Model
    '''
    def __init__(
        self,
        n_dims: int,
        input_keys: list = ["flair", "t1"],
        label_key: str = "WMH",
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        depth: int = 4,
        use_transpose: bool = False,
        use_normalization: bool = True,
        final_activation: nn.Module = nn.Identity(),
        inferer: str = "sliding_window",
        inferer_params: dict = None,
        learning_rate: float = 1e-3,
        loss_fn: Callable = F.mse_loss,
    ) -> None:
        '''
        Initializes the LitUNet Lightning module, a PyTorch Lightning wrapper 
        around a U-Net model for deep learning tasks.
    
        Parameters:
        ----------
        n_dims : int
            The number of spatial dimensions (2 for 2D or 3D data).
        input_keys : list[str]
            List of keys representing input modalities.
        label_key : str
            The key representing the segmentation map.
        in_channels : int, optional
            Number of input channels to the model (default is 1).
        out_channels : int, optional
            Number of output channels from the model (default is 1).
        base_channels : int, optional
            Number of base channels in the U-Net's encoder (default is 8).
        depth : int, optional
            Depth of the U-Net (number of downsampling steps, default is 4).
        use_transpose : bool, optional
            Whether to use transpose convolutions for upsampling 
            (default is False, using interpolation instead).
        use_normalization : bool, optional
            Whether to apply normalization layers in the U-Net (default is True).
        final_activation : nn.Module, optional
            Activation function applied to the model's output (default is nn.Identity).
        inferer : str, optional
            Type of inferer to use: "sliding_window", "slice", or "simple" (default is "sliding_window").
        inferer_params : dict, optional
            Additional parameters for the inferer (default is None).
        learning_rate : float, optional
            Learning rate for the optimizer (default is 1e-3).
        loss_fn : Callable, optional
            Loss function to optimize during training (default is F.mse_loss).
        '''
        super().__init__()
        self.save_hyperparameters()

        # Loss and learning rate 
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        # Input and label keys
        ### Defining a collate_fn in out dataloader might achieve the same result as what we are doing here
        self.input_keys = input_keys
        self.label_key = label_key
        # print(f"input keys: {self.input_keys}") # DEBUG
        # print(f"label key: {self.label_key}") # DEBUG

        # UNet model setup
        self.model = UNet(
            n_dims=n_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            use_transpose=use_transpose,
            use_normalization=use_normalization,
            final_activation=final_activation,
        )

        # Setting Monai inference
        inferer_params = inferer_params or {}
        if inferer.lower() == "sliding_window":
            self.inferer = SlidingWindowInferer(**inferer_params)
        elif inferer.lower() == "slice":
            self.inferer = SliceInferer(**inferer_params)
        elif inferer.lower() == "simple":
            self.inferer = SimpleInferer()
        else:
            raise ValueError(f'"{inferer}" is not a supported inferer type. Use "sliding_window", "slice", or "simple".')

        num_classes = 2 
        # Setting validation and test metrics
        self.validation_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes),
                "precision": torchmetrics.classification.Precision(task="binary", num_classes=num_classes),
                "recall": torchmetrics.classification.Recall(task="binary", num_classes=num_classes),
                "f1": torchmetrics.classification.F1Score(task="binary", num_classes=num_classes)
            },
            prefix="val_",
        )
        
        self.writer = SummaryWriter()

    def forward(self, x):
        return self.model(x)

    def _extract_inputs_and_labels(self, batch, input_keys, label_key):
        """
        Extracts the input modalities and segmentation labels from the batch.
    
        Parameters
        ----------
        batch : dict
            The batch containing the data.
        input_keys : list[str]
            List of keys representing input modalities.
        label_key : str
            The key representing the segmentation map.
    
        Returns
        -------
        x : torch.Tensor
            Concatenated input modalities.
        y : torch.Tensor
            Segmentation map.
        """
        x = torch.cat([batch[key] for key in input_keys], dim=1) 
        y = batch[label_key]
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._extract_inputs_and_labels(batch, self.input_keys, self.label_key)
        y_hat = self.inferer(x, self.model)

        # Ensure y is binary
        # y = (y > 0).int()
        
        # Apply softmax to get probabilities
        y_hat = torch.softmax(y_hat, dim=1) # TRYING OUT SHOULD BE CHANGED LATER
        
        loss = self.loss_fn(y_hat, y)
        # self.log("train_loss", loss, prog_bar=True)

        ######## Copied from validation
        y_pred = (y_hat > 0.5).int()

        # Ensure y is binary
        y = (y > 0).int()
        
        # Ensure y_true is in the correct shape and type
        y_true = y.squeeze(1).long()

        # Flatten tensors for metric computation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate F1 score and recall
        self.log_dict(self.validation_metrics(y_pred, y_true), on_step=True, on_epoch=True)

        ##############
        
        # Log the loss to TensorBoard
        # self.writer.add_scalar("Loss/train", loss, self.global_step)
        
        return loss

    def validation_step(self, batch, batch_idx):# f1 and recall do not work here 
        x, y = self._extract_inputs_and_labels(batch, self.input_keys, self.label_key)
        y_hat = self.inferer(x, self.model)
        val_loss = self.loss_fn(y_hat, y)
        # self.log("val_loss", val_loss, prog_bar=True)
        
        # Log the validation loss to TensorBoard
        # self.writer.add_scalar("Loss/val", val_loss, self.global_step)

        # DEBUG
        # print("BEFORE CONVERSION") 
        # print(f"y_pred shape: {y_hat.shape}")
        # print(f"y_pred: {y_hat}") 
        # print("====================================================") 
        # print(f"y_true shape: {y.shape}") 
        # print(f"y_true unique values: {torch.unique(y)}")
        # print(f"y_true: {y}") 

        # Apply softmax to get probabilities
        y_hat = torch.softmax(y_hat, dim=1) # TRYING OUT SHOULD BE CHANGED LATER
        
        # Convert predictions to binary labels
        ### NOTE: this assumes that the logist are converted to probabilties using nn.SoftMax as the final activation layer
        y_pred = (y_hat > 0.5).int()

        # Ensure y is binary
        y = (y > 0).int()
        
        # Ensure y_true is in the correct shape and type
        y_true = y.squeeze(1).long()

        # Flatten tensors for metric computation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # y_pred = y_hat.argmax(dim=1).cpu().numpy().flatten()
        # y_true = y.cpu().numpy().flatten()
        
        # # Ensure y_true is binary
        # y_true = (y_true > 0).astype(int)

        # DEBUG
        # print("AFTER CONVERSION")
        # print(f"y_pred shape: {y_pred.shape}")
        # print(f"y_pred unique values: {torch.unique(y_pred)}")
        # print(f"y_pred: {y_pred}") # DEBUG
        # print("====================================================")
        # print(f"y_true shape: {y_true.shape}")
        # print(f"y_true: {y_true}") # DEBUG
        
        # Calculate F1 score and recall
        self.log_dict(self.validation_metrics(y_pred, y_true), on_step=True, on_epoch=True)
        
        # val_f1 = f1_score(y_true, y_pred, average='macro')
        # val_recall = recall_score(y_true, y_pred, average='macro')
        
        # Log the metrics to TensorBoard
        # self.writer.add_scalar("F1/val", val_f1, self.global_step)
        # self.writer.add_scalar("Recall/val", val_recall, self.global_step)
        
        # Log the metrics to the progress bar
        # self.log("val_f1", val_f1, prog_bar=True)
        # self.log("val_recall", val_recall, prog_bar=True)

    def test_step(self, batch, batch_idx):# f1 and recall do not work here
        x, y = self._extract_inputs_and_labels(batch, self.input_keys, self.label_key)
        y_hat = self.inferer(x, self.model)
        test_loss = self.loss_fn(y_hat, y)
        self.log("test_loss", test_loss, prog_bar=True)
        
        # Log the test loss to TensorBoard
        self.writer.add_scalar("Loss/test", test_loss, self.global_step)
        
        # Convert predictions to binary labels
        y_pred = y_hat.argmax(dim=1).cpu().numpy().flatten()
        y_true = y.cpu().numpy().flatten()
        
        # Ensure y_true is binary
        y_true = (y_true > 0).astype(int)
        
        # Calculate F1 score and recall
        test_f1 = f1_score(y_true, y_pred, average='macro')
        test_recall = recall_score(y_true, y_pred, average='macro')
        
        # Log the metrics to TensorBoard
        self.writer.add_scalar("F1/test", test_f1, self.global_step)
        self.writer.add_scalar("Recall/test", test_recall, self.global_step)
        
        # Log the metrics to the progress bar
        self.log("test_f1", test_f1, prog_bar=True)
        self.log("test_recall", test_recall, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    
