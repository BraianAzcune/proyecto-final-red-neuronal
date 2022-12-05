from torch import nn
from math import floor


def instanciarModeloConvolucional(config):
    modelo = RedConvolucional(
        cant_filtros_conv1=config["cant_filtros_conv1"],
        kernel_size_maxpool1=config["kernel_size_maxpool1"],
        cant_filtros_conv2=config["cant_filtros_conv2"],
        kernel_size_maxpool2=config["kernel_size_maxpool2"],
        full_l1=config["full_l1"],
        full_l2=config["full_l2"]
    )
    return modelo


class RedConvolucional(nn.Module):

    def __calc_output_shape(self, input_shape, kernel_size, padding=0, stride=1):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nota en la formula "[]" significa funcion floor.
        return floor((input_shape+2*padding-kernel_size)/stride)+1

    def __init__(self, cant_filtros_conv1=6, kernel_size_maxpool1=2, cant_filtros_conv2=16, kernel_size_maxpool2=2, full_l1=120, full_l2=84) -> None:
        super().__init__()
        # se reciben imagenes 3x32x32
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=cant_filtros_conv1,
                      kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_maxpool1, stride=2)
        )
        # [cantidad filtros, dimension de tanto altura como anchura]
        self.o_conv2d = [cant_filtros_conv1, self.__calc_output_shape(32, 5, 1, 1)]

        self.o_maxPool = [cant_filtros_conv1, self.__calc_output_shape(
            self.o_conv2d[1], kernel_size_maxpool1, 0, 2)]

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.o_maxPool[0], out_channels=cant_filtros_conv2,
                      kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_maxpool2, stride=2)
        )

        self.o_conv2d2 = [cant_filtros_conv2,
                     self.__calc_output_shape(self.o_maxPool[1], 5, 0, 1)]

        
        self.o_maxPool2 = [cant_filtros_conv2, self.__calc_output_shape(self.o_conv2d2[1],kernel_size_maxpool2,0,2)]

        self.cantidad_entradas = self.o_maxPool2[0] * pow(self.o_maxPool2[1], 2)

        self.denseLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.cantidad_entradas, out_features=full_l1),
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
