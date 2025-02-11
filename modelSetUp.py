import ast  # Para convertir string de listas a listas reales
import shutil
import yaml
import os
import pandas as pd

# Definir las clases en orden
CLASSES = [
    "Nipple Retraction", "Global Asymmetry", "No Finding", "Asymmetry",
    "Skin Retraction", "Suspicious Calcification", "Focal Asymmetry",
    "Skin Thickening", "Mass", "Architectural Distortion", "Suspicious Lymph Node"
]

# Crear un diccionario para asignar índices a cada categoría
class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

metadata_dir = "pf-bula-castillo-garcia/annotationsv2.csv"
annotations = pd.read_csv(metadata_dir)

def get_class_index(category_str):
    """
    Extrae la clase y la convierte en su índice numérico.
    """
    if isinstance(category_str, list):  # Si ya es una lista, la usamos directamente
        category_list = category_str
    else:
        try:
            category_list = ast.literal_eval(category_str)  # Convertir string a lista
        except (ValueError, SyntaxError):
            return -1  # Si hay un error, devolver -1
    
    if category_list:  # Si la lista no está vacía
        return class_to_idx.get(category_list[0], -1)  # Retorna índice de la clase
    return -1  # Si no tiene categoría

# Aplicar la conversión
annotations["class_id"] = annotations["finding_categories"].apply(get_class_index)



yolo_labels_path = "pf-bula-castillo-garcia/yolo_labels/"
os.makedirs(yolo_labels_path, exist_ok=True)

for i, row in annotations.iterrows():
    img_path = f'{row.directory_path}'
    
    if row.class_id == -1:
        continue
    
    # Normalizar coordenadas para YOLO (Algo raro en google, pruebo primero con las que ya saque como dijo
    # el profe eduardo)

    x_center = row.bx
    y_center = row.by
    bbox_width = row.bw
    bbox_height = row.bh
    
    #x_center = row.bx / row.width
    #y_center = row.by / row.height
    #bbox_width = row.bw / row.width
    #bbox_height = row.bh / row.height

    # Crear archivo de anotaciones YOLO
    yolo_label_path = os.path.join(yolo_labels_path, f"{row.image_id}.txt")
    with open(yolo_label_path, "w") as f:
        f.write(f"{row.class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

base_yolo_path = "pf-bula-castillo-garcia/dataset/"

for i, row in annotations.iterrows():
    split_folder = "train" if row.split == "training" else "val"
    
    # Mover imagen a la carpeta de imágenes YOLO
    img_dest_path = os.path.join(base_yolo_path, "images", split_folder, f"{row.image_id}.png") #cambie jpg a png como se manejaba desde antes las imgs
    os.makedirs(os.path.dirname(img_dest_path), exist_ok=True)
    shutil.copy(row.directory_path, img_dest_path)
    
    # Mover anotaciones a la carpeta de labels YOLO
    label_src_path = os.path.join(yolo_labels_path, f"{row.image_id}.txt")
    label_dest_path = os.path.join(base_yolo_path, "labels", split_folder, f"{row.image_id}.txt")
    os.makedirs(os.path.dirname(label_dest_path), exist_ok=True)
    shutil.copy(label_src_path, label_dest_path)

# Definir la configuración para YOLO
yolo_config = {
    "train": "dataset/images/train",  # Ruta a los datos de entrenamiento
    "val": "dataset/images/val",      # Ruta a los datos de validación
    "nc": 11,  # Número de clases (según las 11 categorías que mencionaste)
    "names": [
        "Nipple Retraction", "Global Asymmetry", "No Finding", "Asymmetry", "Skin Retraction",
        "Suspicious Calcification", "Focal Asymmetry", "Skin Thickening", "Mass",
        "Architectural Distortion", "Suspicious Lymph Node"
    ]
}

# Ruta donde se guardará el archivo
yaml_path = "pf-bula-castillo-garcia/data.yaml"

# Guardar el archivo YAML
with open(yaml_path, "w") as file:
    yaml.dump(yolo_config, file, default_flow_style=False)

print(f"Archivo YAML guardado en: {yaml_path}")


dataset_path = "pf-bula-castillo-garcia/dataset/"
expected_folders = ["images/train", "images/val", "labels/train", "labels/val"]

for folder in expected_folders:
    full_path = os.path.join(dataset_path, folder)
    print(f"{full_path}: {'✅ Existe' if os.path.exists(full_path) else '❌ No existe'}")