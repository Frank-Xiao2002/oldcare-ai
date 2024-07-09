from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# import dataset from Roboflow and you can get the data.yml file
results = model.train(data='data.yaml', imgsz=640, epochs=100, batch=8)
