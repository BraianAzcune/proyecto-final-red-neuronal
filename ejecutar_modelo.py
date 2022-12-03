from typing import List
import os
from datetime import datetime
import csv

def main():
    print("hello world!")









def crear_archivo_estadisticas(path:str, columnas:List[str])->str:
    """
        Returns:
            [str] -- path del archivo creado
    """
    now = datetime.now()
    date_time_str = now.strftime(f"(%Y-%m-%d_%H:%M:%S)")
    name = path + "/estadisticas-" + date_time_str
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columnas)
    return name

def guardar_estadisticas(path:str, datos:List[str]):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(datos)


if __name__ == "__main__":
    path_estadisticas = os.path.abspath("./datos_estadistica/")

    #main()
    dir = crear_archivo_estadisticas(path_estadisticas,["pepe","jose"])
    guardar_estadisticas(dir,["23","12"])
