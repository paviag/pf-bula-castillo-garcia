from ultralytics import YOLO

model = YOLO("yolov8n.pt") # No se si este es el correcto, en este link estan todos los modelos #https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes
model.train(data="pf-bula-castillo-garcia\data.yaml", epochs=5, imgsz=300) #no se si ese imgsz afecte el resultado por el resize automatico que hace

model.export(format="onnx")


#results = model("pf-bula-castillo-garcia\dataset\images\val\0af3e36d8d050e92149a6b79e4181db4.jpg")
#results.show()
