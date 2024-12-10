from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model  # train the model
results = model.train(
   data="/home/ereshkigal/hakathon/mltraining/MLTraining/yolov8/dataset/pothole_dataset_v8/pothole.yaml",
   imgsz=1280,
   epochs=50,
   batch=8,
   name='yolov8n_v8_50e'
)