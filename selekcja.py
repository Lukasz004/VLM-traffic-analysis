import numpy as np
from ultralytics import YOLO
import cv2
import os

MATRIXPATH = "interactionMatrix.npy"
VIDEOPATH = "film.mp4"
OUTPUTPATH = "output/"
THRESHOLD = 0.1

#definicja do jakich grup naleza poszczególne klasy, group1 to pojazdy a group2 to osoby
group1 = ['car', 'truck', 'bus', 'motorcycle']
group2 = ['person', 'umbrella']

os.makedirs(OUTPUTPATH, exist_ok=True)

#ladowanie macierzy interakcji, modelu detekcji i strumienia wideo
matrix = np.load(MATRIXPATH)
model = YOLO("yolo26n.pt")

video = cv2.VideoCapture(VIDEOPATH)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

recordingID = 0
out = None
framesLeft = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    results = model(frame, verbose=False)
    
    triggerGroup1 = False
    triggerGroup2 = False
    
    #analiza wykrytych obiektów pod kątem obecności w strefie
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
        classID = int(result.cls[0].item())
        className = model.names[classID]
        
        #zabezpieczenie przed wyjściem bboxa poza kadr
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)

        #sprawdzenie, czy którykolwiek róg bboxa nachodzi strefę interakcji
        if (matrix[y1, x1] > THRESHOLD or 
            matrix[y2, x1] > THRESHOLD or 
            matrix[y1, x2] > THRESHOLD or 
            matrix[y2, x2] > THRESHOLD):
            
            if className in group1:
                triggerGroup1 = True
            elif className in group2:
                triggerGroup2 = True

    #inicjalizacja zapisu, jeśli wykryto obiekty z obu grup w strefie
    if triggerGroup1 and triggerGroup2:
        framesLeft = 150 #ustawienie bufora klatek po zaniku zdarzenia
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f"{OUTPUTPATH}output{recordingID}.mp4", fourcc, fps, (width, height))

    #logika podtrzymania nagrywania
    if framesLeft > 0:
        out.write(frame)
        
        if not (triggerGroup1 and triggerGroup2):
            framesLeft -= 1
            
        if framesLeft == 0:
            out.release()
            out = None
            recordingID += 1

if out is not None:
    out.release()
    
video.release()
cv2.destroyAllWindows()