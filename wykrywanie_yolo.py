import cv2
from ultralytics import YOLO

#zmienne globalne do rysowania linii
line_pts = []

def select_line(event, x, y, flags, param):
    global line_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_pts) < 2:
            line_pts.append((x, y))

def get_side(p1, p2, pt):
    #iloczyn wektorowy
    #zwraca +1 lub -1 w zależności od tego, po której stronie linii znajduje się punkt, 0 oznacza współliniowość.
    d = (pt[0] - p1[0]) * (p2[1] - p1[1]) - (pt[1] - p1[1]) * (p2[0] - p1[0])
    if d > 0: return 1
    elif d < 0: return -1
    else: return 0

def main():
    global line_pts
    cap = cv2.VideoCapture("video.mp4")
    model = YOLO('yolo26n.pt')

    ret, frame = cap.read()

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", select_line)

    print("-----------------------------------------------------")
    print("KLIKNIJ dwa razy na obrazie kamery, aby narysować linię.")
    print("Następnie wciśnij klawisz 'SPACJA', aby uruchomić detekcję.")
    print("-----------------------------------------------------")

    while True:
        temp_frame = frame.copy()
        
        for pt in line_pts:
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
        if len(line_pts) == 2:
            cv2.line(temp_frame, line_pts[0], line_pts[1], (0, 255, 0), 2)
            cv2.putText(temp_frame, "Wcisnij SPACJE aby kontynuowac", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Video", temp_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if len(line_pts) == 2 and key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    p1, p2 = line_pts[0], line_pts[1]
    
    tracked_sides = {} 
    crossed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #wbudowany tracker YOLO automatycznie przypisuje i utrzymuje ID
        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            #rzutujemy tensory na inne typy aby mozna bylo potem wykorzystac wartosci
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_side = get_side(p1, p2, (cx, cy))

                if track_id in tracked_sides:
                    old_side = tracked_sides[track_id]
                    
                    #detekcja przejścia przez linie, obecny znak różni się od poprzedniego i obeikt nie jest na 0
                    if old_side != 0 and current_side != 0 and old_side != current_side:
                        crossed_count += 1
                        print(f"Obiekt ID:{track_id} przekroczył linię! (Suma: {crossed_count})")
                        
                        tracked_sides[track_id] = current_side
                else:
                    tracked_sides[track_id] = current_side

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        cv2.line(frame, p1, p2, (0, 0, 255), 3)
        cv2.putText(frame, f"Przekroczenia: {crossed_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()