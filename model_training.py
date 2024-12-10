import torch
from tqdm.auto import tqdm
from typing import Callable

class ModelTraining():
    def __init__(
        self,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        loss_fn: Callable,
        samples: list
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_dims=2, in_channels=1, out_channels=1, depth=3).to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_ds, self.train_dl, self.val_ds, self.val_dl = self._split_samples(samples)
    
    def run_training_loop(self):
        for epoch in (prog_bar := tqdm(range(self.n_epochs), desc="Training", unit="epoch", total=self.n_epochs, position=0)):
            prog_bar.set_description(f"Training Loop")
            train_losses = self._loop_train()
            prog_bar.set_postfix({"Training loss": sum(train_losses) / len(train_losses)})
            prog_bar.set_description(f"Validation Loop")
            val_losses = self._loop_validate()
            prog_bar.set_postfix({"Training loss": sum(train_losses) / len(train_losses), "Validation loss": sum(val_losses) / len(val_losses)})
        
    def _loop_train(self):
        self.model.train()
        train_losses = []

        for i, (image, label) in tqdm(enumerate(self.train_dl), total=len(self.train_dl), desc="Training", unit="batch", position=1, leave=False):
            
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad() # Clear gradients
            output = self.model(...) # Model forward pass
            loss = loss_fn(output, label)  # Compute loss
            loss.backward()  # Backpropagate loss
            self.optimizer.step()  # Update model weights

            train_losses.append(loss.item()) # Append training loss for this batch

        return train_losses

    def _loop_validate(self):
        self.model.eval() # We set the model in evaluation mode
        val_losses = []
        for i, (image, label) in tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc="Validation", unit="batch", position=1, leave=False):
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                loss = loss_fn(output, label)
            
            val_losses.append(loss.item())

        return val_losses

    def _split_samples(self, samples): #TODO - adjust this function to our data
        train_samples = samples[0 : int(len(samples) * 0.8)]
        val_samples = samples[int(len(samples) * 0.8): ]

        train_ds = MedicalDecathlonDataset(train_samples) #TODO - change to our own dataset class
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True) #TODO - change to our DataLoader class

        val_ds = MedicalDecathlonDataset(val_samples) #TODO - change to our own dataset class
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False) #TODO - change to our DataLoader class
        return train_ds, train_dl, val_ds, val_dl
