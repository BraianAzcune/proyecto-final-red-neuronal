from numpy import pad
import torch
from torch import nn

"""
TODO hay que revisar entre Lenet5 y AlexNet, las diferencias entre ambas, y asi crear un rango de busqueda para los hiperparametros que definen la estructura interna de la red.
"""

class RedConvolucional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # se reciben imagenes 3x32x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0), # quedan un volumen de 6x28x28 # (n - k + s + p)/s
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # queda un volumen de 6x14x14
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), # queda un volumen de 16x10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # queda un volumen de 16x5x5 = 400.
        )

        self.denseLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )
    
    def forward(self, x):
        logits = self.layer1(x)
        logits = self.layer2(logits)
        logits = self.denseLayer(logits)
        # no es necesario utilizar softmax, ya que el optimizador lo integra, pero es necesario agregarlo por afuera para cuando se utilice el modo de evaluacion.
        return logits