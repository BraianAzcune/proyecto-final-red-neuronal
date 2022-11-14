from numpy import pad
import torch
from torch import nn
from math import floor

"""
TODO hay que revisar entre Lenet5 y AlexNet, las diferencias entre ambas, y asi crear un rango de busqueda para los hiperparametros que definen la estructura interna de la red.
"""


class RedConvolucional(nn.Module):
    def __init__(self, cant_filtros_conv1=6, kernel_size_maxpool1=2, cant_filtros_conv2=16, kernel_size_maxpool2=2,full_l1=120,full_l2=84) -> None:
        super().__init__()
        # se reciben imagenes 3x32x32
        self.layer1 = nn.Sequential(
            # quedan un volumen de cant_filtros_conv1x30x30 # (n - k + s + p)/s
            nn.Conv2d(in_channels=3, out_channels=cant_filtros_conv1,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # queda un volumen de cant_filtros_conv1 x 15 x 15 (si kernel_size_maxpool1=2, si es 3, entonces= 14)
            nn.MaxPool2d(kernel_size=kernel_size_maxpool1, stride=2)
        )

        o_conv2d = [cant_filtros_conv1, 32-5+1+2] # 30
        o_maxPool = [cant_filtros_conv1, floor((o_conv2d[1] - kernel_size_maxpool1 + 2)/2)]

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=o_maxPool[0], out_channels=cant_filtros_conv2,
                      kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_maxpool2, stride=2)
        )

        o_conv2d2 = [cant_filtros_conv2, o_maxPool[1] - 5 + 1]
        o_maxPool2 = [cant_filtros_conv2, floor((o_conv2d2[1]-kernel_size_maxpool2+2)/2)]

        cantidad_entradas = floor(o_maxPool2[0] * pow(o_maxPool2[1],2))

        self.denseLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=cantidad_entradas, out_features=full_l1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=full_l1, out_features=full_l2),
            nn.ReLU(),
            nn.Linear(in_features=full_l2, out_features=10)
        )

    def forward(self, x):
        logits = self.layer1(x)
        logits = self.layer2(logits)
        logits = self.denseLayer(logits)
        # no es necesario utilizar softmax, ya que el optimizador lo integra, pero es necesario agregarlo por afuera para cuando se utilice el modo de evaluacion.
        return logits
