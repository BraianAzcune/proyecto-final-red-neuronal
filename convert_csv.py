import re
def convert_console_table_to_csv(file_path, separator=";"):
    respuesta = ""
    with open(file_path,"r") as f:
        # ignorar primera linea cabecera
        f.readline()
        # segunda linea tiene los datos de cabecera
        
        for linea in f:
            if linea[1]=="-": continue
            rta = convert_linea(linea,separator)
            respuesta += rta
    
    
    with open(file_path+".csv","w") as f:
        f.write(respuesta)



def convert_linea(rough_linea, separator):
    # separar, y remover primer elemento, no sirve.
    rough_linea=rough_linea.split("|")[1:]
    # eliminar espacios inecesarios.
    rough_linea = [re.sub("\s\s+", " ", x) for x in rough_linea]
    linea = separator.join(rough_linea)
    return linea

if __name__ == "__main__":
    import os
    file_path = os.path.abspath("./output_experiment.txt")
    convert_console_table_to_csv(file_path=file_path)
