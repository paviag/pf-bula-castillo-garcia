from ultralytics import YOLO

model = YOLO("yolov8n.pt") # No se si este es el correcto, en este link estan todos los modelos #https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes
model.train(data="/kaggle/working/dataset/data.yaml", epochs=5, imgsz=300) #no se si ese imgsz afecte el resultado por el resize automatico que hace


results = model("/kaggle/working/dataset/images/val/ejemplo.jpg")
results.show()
