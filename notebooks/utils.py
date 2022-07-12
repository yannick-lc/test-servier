import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(
    train_losses: np.ndarray, val_losses: np.ndarray, val_aucs: np.ndarray,
) -> None:
    """
    Plot the evolution of the train losses, val losses and val AUCs
    obtained during the training of a deep learning model on a single figure.
    """
    n_epochs = train_losses.shape[0]
    epochs = np.arange(n_epochs)
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    plt.plot(epochs, train_losses)
    plt.title("Evolution of training loss");plt.xlabel("Epoch");plt.ylabel("Training loss")
    plt.subplot(1,3,2)
    plt.plot(epochs, val_losses)
    plt.title("Evolution of validation loss");plt.xlabel("Epoch");plt.ylabel("Validation loss")
    plt.subplot(1,3,3)
    plt.plot(epochs, val_aucs)
    plt.title("Evolution of validation AUC");plt.xlabel("Epoch");plt.ylabel("Validation AUC")
    plt.show()