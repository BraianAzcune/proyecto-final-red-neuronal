import os
import csv
from typing import List
from datetime import datetime
from modelo_convolucional import instanciarModeloConvolucional
from modelo_mlp import instanciarModeloMLP
from cargar_datos import cargar_datasets, cargar_dataloaders
# import pytorch
import torch
from torch import nn
from torch.utils.data import random_split


def main(model, config, checkpoint_dir: str, path_datos_entrenamiento: str, num_epochs: int, path_estadisticas: str, model_state=None, optimizer_state=None, last_epoch=0):

    if model_state is not None:
        # restaurar estado modelo anterior
        model.load_state_dict(model_state)

    device = getDevice()
    # mover el procesamiento a gpu si existe.
    model.to(device)

    # CrossEntropyLoss requiere el output "logits", no es necesario pasarlo por el softmax, ya que lo calcula dentro. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()

    # se necesita enviar los parametros del modelo al optimizador para que los pueda actualizar.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    if optimizer_state is not None:
        # restaurar estado optimizador anterior
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = cargar_datasets(path_datos_entrenamiento)
    # reservar una parte de los datos de entrenamiento para validacion.
    train_subset, val_subset = partir_data_set(trainset, 0.2)

    # los dataloaders, convierten los datos ya para ser usado por el modelo, y se peden iterar entregando
    # batches de datos, ademas de mezclarlos.
    trainloader = cargar_dataloaders(train_subset, config["batch_size"])
    valloader = cargar_dataloaders(val_subset, config["batch_size"])
    testloader = cargar_dataloaders(testset, config["batch_size"])

    # crear csv de estadistica
    path_archivo_csv = crear_archivo_estadisticas(
        path_estadisticas, ["epoch", "lossTrain", "lossVal", "accuracyTrain", "accuracyVal", "accuracyTest"])

    # loop entrenar validar, guardar copia y estadistica.
    for epoch in range(last_epoch+1, last_epoch+num_epochs+1):
        lossTrain, accuracyTrain = train_loop(
            model, device, trainloader, criterion, optimizer)
        lossVal, accuracyVal = validation_loss(
            model, device, valloader, criterion)
        accuracyTest = test_accuracy(model, device, testloader)

        print("epoch", epoch)
        print("lossTrain", "accuracyTrain")
        print(lossTrain, accuracyTrain)
        print("lossVal", "accuracyVal")
        print(lossVal, accuracyVal)
        print("accuracyTest", accuracyTest)

        guardar_estadisticas(path_archivo_csv, [epoch,
                             lossTrain, lossVal, accuracyTrain, accuracyVal, accuracyTest])
        # guardar modelo cada 10 epocas.
        if epoch % 10 == 9:
            guardar_estado_modelo(checkpoint_dir, epoch, model, optimizer)


def guardar_estado_modelo(checkpoint_dir, epoch, model, optimizer):
    name = checkpoint_dir + "/modelo_checkpoint"+str(epoch)
    with open(name, "w") as f:
        torch.save((model.state_dict(), optimizer.state_dict()), name)


def train_loop(model, device, trainloader, criterion, optimizer):
    # variables para calcular loss y accuraccy
    train_loss = 0.0
    train_steps = 0
    total = 0
    correct = 0

    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # calcular loss y accuraccy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loss += loss.cpu().detach().numpy()
        train_steps += 1

        # backward pass y optimzacion
        loss.backward()
        optimizer.step()
    loss = train_loss / train_steps
    accuracy = correct / total
    return loss, accuracy


def validation_loss(model, device, valloader, criterion):
    # variables para calcular loss y accuraccy
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    # pone a la red en moodo evaluacion, desactiva capas dropout.
    model.eval()
    for i, data in enumerate(valloader, 0):
        # desactiva el proceso de calculo y guardado de valores intermedios
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # calcular loss y accuraccy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.cpu().numpy()
            val_steps += 1

    loss = val_loss / val_steps
    accuracy = correct / total
    return loss, accuracy


def test_accuracy(model, device, testloader):
    correct = 0
    total = 0
    # pone a la red en moodo evaluacion, desactiva capas dropout.
    model.eval()
    # desactiva el proceso de calculo y guardado de valores intermedios
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def partir_data_set(data_set, porcentaje_validacion: float):
    """
    Args:
        porcentaje_validacion(float): si se quiere reservar 10% para validacion hay que enviar 0.1
    Returns:
        dataset de entramiento con una cantidad (1 - porcentaje_validacion) %
        dataset de validacion con una cantidad porcentaje_validacion %
    """
    test_abs = int(len(data_set) * (1-porcentaje_validacion))
    train_subset, val_subset = random_split(
        data_set, [test_abs, len(data_set) - test_abs])
    return train_subset, val_subset


def getDevice():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            raise Exception(
                "la pc cuenta con multiples gpus, deberia utilizar DataParallel")
            #model = nn.DataParallel(model)
    return device


def crear_archivo_estadisticas(path: str, columnas: List[str]) -> str:
    """
        Returns:
            [str] -- path del archivo creado
    """
    now = datetime.now()
    date_time_str = now.strftime(f"(%Y-%m-%d_%H:%M:%S)")
    name = path + "/estadisticas-" + date_time_str + ".csv"
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columnas)
    return name


def guardar_estadisticas(path: str, datos: List[str]):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(datos)


def cargar_estado_modelo_optimizador(path):
    model_state, optimizer_state = torch.load(path)
    return model_state, optimizer_state


if __name__ == "__main__":
    checkpoint_dir = os.path.abspath("./checkpoint_dir")
    path_estadisticas = os.path.abspath("./datos_estadistica/")
    path_datos_entrenamiento = os.path.abspath("./data")

    ultimo_checkpoint = "0"
    model_state = None
    optimizer_state = None

    # checkpoint_state = os.path.join(
    #     checkpoint_dir, "modelo_checkpoint"+ultimo_checkpoint)
    # model_state, optimizer_state = cargar_estado_modelo_optimizador(
    #     checkpoint_state)

    # Configuracion del modelo y instanciarlo
    config = {
        "batch_size": 16,
        "learning_rate": 0.000106647,
        "cant_filtros_conv1": 18,
        "kernel_size_maxpool1": 2,
        "cant_filtros_conv2": 28,
        "kernel_size_maxpool2": 3,
        "full_l1": 160,
        "full_l2": 104,
        "weight_decay":1e-4,
    }
    # instanciar modelo convolucional
    #modelo = instanciarModeloConvolucional(config)
    modelo = instanciarModeloMLP(config)

    main(model=modelo,
         config=config,
         checkpoint_dir=checkpoint_dir,
         num_epochs=200,
         path_datos_entrenamiento=path_datos_entrenamiento,
         path_estadisticas=path_estadisticas,
         model_state=model_state,
         optimizer_state=optimizer_state,
         last_epoch=int(ultimo_checkpoint)
         )
    # dir = crear_archivo_estadisticas(path_estadisticas,["pepe","jose"])
    # guardar_estadisticas(dir,["23","12"])
