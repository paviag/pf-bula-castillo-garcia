from ultralytics import YOLO

model_path = "C:/Users/Lab6k/Documents/PF/runs/detect/train242/weights/best.pt"
test_path = ""

model = YOLO(model_path)

print(model.val().box.map)
results = model("pf-bula-castillo-garcia/dataset/images/val/0af3e36d8d050e92149a6b79e4181db4.jpg")
results.show()
