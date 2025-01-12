import torch
from tqdm.auto import tqdm
from typing import Callable
from torch.utils.tensorboard import SummaryWriter
from model import LitUNet  # Assuming your model class is named LitUNet
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm
import torchmetrics

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
        self.model = LitUNet().to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_ds, self.train_dl, self.val_ds, self.val_dl = self._split_samples(samples)
        self.writer = SummaryWriter()
        self.n_epochs = n_epochs
    
    def run_training_loop(self):
        for epoch in (prog_bar := tqdm(range(self.n_epochs), desc="Training", unit="epoch", total=self.n_epochs, position=0)):
            prog_bar.set_description(f"Training Loop")
            train_losses = self._loop_train(epoch)
            prog_bar.set_postfix({"Training loss": sum(train_losses) / len(train_losses)})
            prog_bar.set_description(f"Validation Loop")
            val_losses = self._loop_validate(epoch)
            prog_bar.set_postfix({"Training loss": sum(train_losses) / len(train_losses), "Validation loss": sum(val_losses) / len(val_losses)})
        self.writer.close()
        
    def _loop_train(self, epoch):
        self.model.train()
        train_losses = []

        for i, (image, label) in tqdm(enumerate(self.train_dl), total=len(self.train_dl), desc="Training", unit="batch", position=1, leave=False):
            
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad() # Clear gradients
            output = self.model(image) # Model forward pass
            loss = self.loss_fn(output, label)  # Compute loss
            loss.backward()  # Backpropagate loss
            self.optimizer.step()  # Update model weights

            train_losses.append(loss.item()) # Append training loss for this batch
            self.writer.add_scalar("Loss/train", loss.item(), epoch * len(self.train_dl) + i)

        return train_losses

    def _loop_validate(self, epoch): # f1 and recall do not work here
        self.model.eval() # We set the model in evaluation mode
        val_losses = []
        all_y_true = []
        all_y_pred = []

        # Initialize metrics
        val_f1 = torchmetrics.F1Score(task='binary', average='macro').to(self.device)
        val_recall = torchmetrics.Recall(task='binary', average='macro').to(self.device)

        for i, (image, label) in tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc="Validation", unit="batch", position=1, leave=False):
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                loss = self.loss_fn(output, label)
            
            val_losses.append(loss.item())

            # Convert predictions to binary labels
            y_pred = output.argmax(dim=1)
            y_true = (label > 0).int().squeeze(1)

            # Update metrics
            val_f1.update(y_pred, y_true)
            val_recall.update(y_pred, y_true)

            # Collect all predictions and labels for further analysis if needed
            all_y_true.extend(y_true.cpu().numpy().flatten())
            all_y_pred.extend(y_pred.cpu().numpy().flatten())

        # Compute final metrics
        final_val_f1 = val_f1.compute().item()
        final_val_recall = val_recall.compute().item()

        print(f"Validation F1 Score: {final_val_f1}")
        print(f"Validation Recall: {final_val_recall}")

        # Optionally, reset metrics for the next epoch
        val_f1.reset()
        val_recall.reset()

        val_loss = sum(val_losses) / len(val_losses)
        self.writer.add_scalar("Loss/val", val_loss, epoch)

        # Log the metrics to TensorBoard
        self.writer.add_scalar("F1/val", final_val_f1, epoch)
        self.writer.add_scalar("Recall/val", final_val_recall, epoch)

        return val_losses

    def _split_samples(self, samples): #TODO - adjust this function to our data
        train_samples = samples[0 : int(len(samples) * 0.8)]
        val_samples = samples[int(len(samples) * 0.8): ]

        train_ds = MedicalDecathlonDataset(train_samples) #TODO - change to our own dataset class
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True) #TODO - change to our DataLoader class

        val_ds = MedicalDecathlonDataset(val_samples) #TODO - change to our own dataset class
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False) #TODO - change to our DataLoader class
        return train_ds, train_dl, val_ds, val_dl
