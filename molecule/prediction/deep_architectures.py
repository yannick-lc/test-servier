"""
Contains the definitions of the deep learning architectures of the models used.
"""

import torch
from torch import nn

class ModelMorgan(nn.Module):
    """
    Architecture of deep learning model 1, based on morgan fingerprints.
    This is a 3-layer fully connected neural net.
    """
    # Default training parameters
    optimizer = torch.optim.SGD
    learning_rate = 1e-2
    batch_size = 100
    n_epochs = 100

    def __init__(self, dim_hidden=1024, dropout_rate=0.5):
        """
        dim_hidden: dimension of the two hidden layers
        dropout_rate: probability to randomly drop a neuron in the layers in which
        dropout is activated
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass on the input tensor
            - input shape: (batch_size, dim_input)
            (dim_input is assumed to be the Morgan fingerprints, so dim_input=2048)
            - output shape: (batch_size, n_classes)
            (this is a binary classification problem, so n_classes=2)
        """
        return self.linear_relu_stack(x)


class ModelSmile(nn.Module):
    """
    Architecture of deep learning model 2, based on SMILE text representation (as bag-of-words)
    This is a 2-layer bidirectional LSTM followed by a fully-connected layer.
    """
    # Default training parameters
    optimizer = torch.optim.Adam
    learning_rate = 1e-3
    batch_size = 100
    n_epochs = 15

    def __init__(self, dim_lstm=64, vocab_size=29, dropout_rate=0.5):
        """
        lstm_size: dimension of the hidden state of the LSTM
        vocab_size: number of words (distinct characters) in the vocabulary,
        determines the dimension of the input
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=dim_lstm,
                            num_layers=3, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(dim_lstm, 2)
        self.activation = nn.ReLU()
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass on the input tensor
            - input shape: (batch_size, sequence_length, vocab_size)
            (on this dataset, sequence_length<100 and vocab_size=29)
            - output shape: (batch_size, n_classes)
            (with n_classes=2)
        """
        _, (h_n, c_n) = self.lstm(x)
        last_state = c_n[-1]
        res = self.activation(last_state)
        res = self.fc(res)
        return res
