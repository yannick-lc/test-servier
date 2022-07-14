import logging
from typing import List, Tuple, Optional
from enum import Enum

from tqdm import tqdm
import numpy as np
import sklearn
import torch
from torch.utils.data import TensorDataset, DataLoader

from molecule.predictors.deep_architectures import ModelMorgan, ModelSmile


class FeatureType(Enum):
    """
    Represents choices for type of features (and therefore model architecture)
    """
    MORGAN = 1
    SMILE = 2


class Model:
    """
    Deep learning model
    """

    def __init__(self, feature_type=FeatureType.MORGAN, state_dict: str=None):
        """
        Load architecture corresponding to morgan fingerprints or SMILEs representation
        based on feature_type,
        and load weights of pre-trained model if state_dict (path to zip file) is specified.
        """
        torch.manual_seed(42) # setting random seed for reproducibility
        if feature_type == FeatureType.MORGAN:
            self.net = ModelMorgan()
        else:
            self.net = ModelSmile()
        if state_dict is not None:
            logging.info(f"Loading state dict located at {state_dict}...")
            self.net.load_state_dict(torch.load(state_dict))
        self._load_training_config()

        if torch.cuda.is_available():
            self.device = "cuda"
            logging.info("Using CUDA for training and inference.")
        else:
            self.device = "cpu"
            logging.warning("CUDA is not available, using CPU only (may be slow during training).")
        self.net.to(self.device)

    def _load_training_config(self) -> None:
        """
        Load training settings based on model architecture
        """
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = self.net.optimizer(self.net.parameters(), self.net.learning_rate)
        self.BATCH_SIZE = self.net.batch_size
        self.EPOCHS = self.net.n_epochs

    def fit(self, X: np.ndarray, y: np.ndarray, validation_set_ratio=0.) -> Optional[Tuple[np.ndarray]]:
        """
        Fit model on provided training data and labels
        and return training losses, validation losses and val ROC AUCs if validation_set_ratio > 0
        """
        if validation_set_ratio > 0.:
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y,
                        test_size=validation_set_ratio, random_state=42)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
        else:
            X_train, y_train = X, y

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE)

        train_losses, val_losses, val_aucs = [], [], []

        for _ in tqdm(range(self.EPOCHS)):
            # train model
            self.net.train() # activate dropout etc
            loss_epoch = self._train_epoch(dataloader)
            train_losses.append(loss_epoch)
    
            # measure performance evolution on val dataset if specified
            if validation_set_ratio > 0:
                self.net.eval() # deactivate dropout etc
                preds_val = self.net(X_val_tensor)
                val_loss = self.loss_fn(preds_val, y_val_tensor).item() # measure val loss
                val_losses.append(val_loss)
                preds_proba = preds_val.detach().cpu().numpy()[:,1]
                val_auc = sklearn.metrics.roc_auc_score(y_val, preds_proba) # measure val AUC
                val_aucs.append(val_auc)

        # if we specified a validation size, we return train and val losses as well as AUCs
        if validation_set_ratio > 0:
            return np.array(train_losses), np.array(val_losses), np.array(val_aucs)


    def _train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train model for 1 epoch and return average loss
        """
        batch_losses = []

        for X, y in dataloader: # iterate over batches
            X, y = X.to(self.device), y.to(self.device) # load batch on GPU if available

            # make predictions and compute loss
            preds = self.net(X)
            loss = self.loss_fn(preds, y)
            batch_losses.append(loss.item())

            # backpropagate and perform 1 step of stochastic gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        mean_batch_loss = np.array(batch_losses).mean()
        return mean_batch_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities between 0 and 1
        """
        self.net.eval() # deactivate dropout etc
        predicted_activations = self.net(torch.FloatTensor(X).to(self.device))
        predicted_probas = torch.nn.Softmax(dim=1)(predicted_activations)[:,1]
        return predicted_probas.detach().cpu().numpy()

    def predict(self, X: np.ndarray, decision_threshold: float=0.5) -> np.ndarray:
        """
        Predict classes (class 1 if probability >= decision_threshold, class 0 otherwise)
        """
        predicted_proba = self.predict_proba(X)
        predicted_class = (predicted_proba >= decision_threshold).astype(int)
        return predicted_class