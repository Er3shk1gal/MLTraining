import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('/home/ereshkigal/hakathon/mltraining/MLTraining/yolov8/runs/detect/yolov8n_v8_50e/weights/best.pt')  # You can use 'yolov8s.pt', 'yolov8m.pt', etc. for different sizes

# Load the image
image_path = '/home/ereshkigal/hakathon/mltraining/MLTraining/yolov8/dataset/pothole_dataset_v8/valid/images/G0011475.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Process results
for result in results:
    boxes = result.boxes.data.tolist()  # Get the bounding boxes
    for result in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.1:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, "patchhola", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Show the image with detections
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()