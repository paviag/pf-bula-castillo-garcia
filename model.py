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

    model.train(data="pf-bula-castillo-garcia\data.yaml", epochs=128, imgsz=640, device=0, workers=1) #no se si ese imgsz afecte el resultado por el resize automatico que hace
    

if __name__ == "__main__":
    main()



# model.export()

# model.train()
# #results = model("pf-bula-castillo-garcia\dataset\images\val\0af3e36d8d050e92149a6b79e4181db4.jpg")
# #results.show()
