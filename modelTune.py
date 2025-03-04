#Probando el tuning
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())  # Debe imprimir True si CUDA está disponible
print(torch.cuda.device_count())  # Número de GPUs detectadas
print(torch.version.cuda)  # Versión de CUDA soportada por PyTorch



def main():
    model = YOLO("yolov8n.pt") # No se si este es el correcto, en este link estan todos los modelos #https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes
    print(model.device)
    gpu0 = torch.device("cuda:0")
    model.to(gpu0)
    print(model.device)

    # Define search space
    search_space = {  # key: (min, max, gain(optional))
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            "shear": (0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (0.0, 1.0),  # image mixup (probability)
            "mixup": (0.0, 1.0),  # image mixup (probability)
            "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
        }
    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        data="pf-bula-castillo-garcia\data.yaml",
        device=0,
        epochs=30,
        iterations=2,
        #optimizer="AdamW",
        space=search_space,
        plots=True,
        save=True,
        val=True,
        workers=1,
    )
    



    

if __name__ == "__main__":
    main()