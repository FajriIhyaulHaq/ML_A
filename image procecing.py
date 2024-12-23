import cv2
import numpy as np

# Parameter konfigurasi
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco'

# Membaca nama kelas dari file
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Menentukan model YOLO
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_cat = False
    found_bird = False
    found_banana = False
    found_person = False
    
    # Menggunakan output dari YOLO untuk deteksi objek
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Memproses hasil deteksi objek
    for i in indices.flatten():  # Gunakan .flatten() untuk menghindari masalah dengan indexing
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        if classNames[classIds[i]] == 'person':
            found_bird = True
        elif classNames[classIds[i]] == 'banana':
            found_banana = True
        elif classNames[classIds[i]] == 'cat':
            found_cat = True

        if classNames[classIds[i]] == 'bird':
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            print('bird')
        
        if classNames[classIds[i]] == 'cat':
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            print('cat')
        
        if classNames[classIds[i]] == 'banana':
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            print('banana')
        if classNames[classIds[i]] == 'person':
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            print('anomali')
        
        if found_cat and found_bird and found_banana:
            print('alert')

# Membuka kamera lokal (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, im = cap.read()
    
    if not ret:
        print("Gagal membaca frame")
        break

    # Mengubah gambar untuk YOLO
    blob = cv2.dnn.blobFromImage(im, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Mendapatkan nama layer output dari model
    layernames = net.getLayerNames()

    # Memperbaiki cara mengakses unconnected output layers
    unconnected_out_layers = net.getUnconnectedOutLayers()

    # Hanya menggunakan langsung indeks unconnected layers
    outputNames = [layernames[i - 1] for i in unconnected_out_layers]

    # Melakukan deteksi dengan YOLO
    outputs = net.forward(outputNames)

    # Memanggil fungsi untuk menemukan objek
    findObject(outputs, im)

    # Menampilkan hasil deteksi
    cv2.imshow('Image', im)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan dan menutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
