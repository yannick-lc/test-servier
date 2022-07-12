import torch
import torch.nn as nn

class ModelMorgan(nn.Module):
    """
    Architecture of deep learning model 1, based on morgan fingerprints
    """
    # Default training parameters
    optimizer = torch.optim.SGD
    learning_rate = 1e-2
    batch_size = 100
    n_epochs = 100

    def __init__(self, dropout_rate=0.5, dim_hidden=1024):
        super(ModelMorgan, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, 2)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class ModelSmile(nn.Module):
    """
    Architecture of deep learning model 2, based on SMILE text representation (as bag-of-words)
    """
    pass