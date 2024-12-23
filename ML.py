import cv2
import numpy as np
import torch
from torchvision import transforms
from models.yolo import YOLOv3  # Pastikan Anda memiliki model YOLO yang diimplementasikan

# Load Model
model = YOLOv3(weights="yolov3.cfg")  # Ubah path dengan model YOLO yang Anda gunakan
model.eval()

# Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Tambahkan dimensi batch

# Load Video
video_path = "cctv_video.mp4"  # Ubah dengan path video CCTV Anda
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Loop untuk membaca frame dari video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video selesai atau error saat membaca frame")
        break

    # Preprocess frame
    input_tensor = preprocess_image(frame)

    # Inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Postprocessing dan visualisasi
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred  # Koordinat dan confidence prediksi
        if conf > 0.5:  # Confidence threshold
            label = f"{model.classes[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow('Deteksi Objek', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()