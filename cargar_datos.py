import torch
import torchvision
import torchvision.transforms as transforms

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def cargar_datasets(directorio="./data"):
    """
    carga los datos de entrenamiento del directorio especificado default=."/data"
    retorna el conjunto de datos de entrenamiento y prueba, ya procesado a tensores con sus escalares en valor entre 0 y 1.
    """
    # transform.ToTensor convierte imagenes 0-255 a 0.-1. https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    transform = transforms.Compose(
        [transforms.ToTensor(),
         ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    trainset = torchvision.datasets.CIFAR10(root=directorio, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=directorio, train=False,
                                       download=True, transform=transform)
    return trainset,testset


def cargar_dataloaders(dataset,batch_size=32):
    """
    genera un loader que mezcla los datos y se puede iterar en lotes de batch_size = 32, como default.
    retorna el dataloader.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)
    return loader

