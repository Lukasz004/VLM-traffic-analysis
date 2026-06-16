import cv2
import math
import numpy as np

line_pts = []

def select_line(event, x, y, flags, param):
    global line_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_pts) < 2:
            line_pts.append((x, y))

def get_signed_distance(p1, p2, pt):
    num = (pt[0] - p1[0]) * (p2[1] - p1[1]) - (pt[1] - p1[1]) * (p2[0] - p1[0])
    den = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if den == 0:
        return 0
    return num / den

def main():
    global line_pts
    cap = cv2.VideoCapture("video.mp4")
    
    cascade_path = "cars.xml"
    detector = cv2.CascadeClassifier(cascade_path)

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
    
    #obliczanie wektora normalnego
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length # wektor prostopadły
    
    tracked_objects = {}
    next_obj_id = 0
    crossed_count = 0
    
    max_distance = 100      #maksymalny przeskok centroidu między klatkami
    crossing_margin = 40   #jak daleko od linii musi odejść obiekt by zliczyć przekroczenie
    
    m1_p1 = (int(p1[0] + nx * crossing_margin), int(p1[1] + ny * crossing_margin))
    m1_p2 = (int(p2[0] + nx * crossing_margin), int(p2[1] + ny * crossing_margin))
    m2_p1 = (int(p1[0] - nx * crossing_margin), int(p1[1] - ny * crossing_margin))
    m2_p2 = (int(p2[0] - nx * crossing_margin), int(p2[1] - ny * crossing_margin))

    #rownoleglobok buforowy
    paralellogram_plus = np.array([p1, p2, m1_p2, m1_p1], dtype=np.int32)
    paralellogram_minus = np.array([p1, p2, m2_p2, m2_p1], dtype=np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        current_centroids = []
        for (x, y, w, h) in detections:
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy, x, y, w, h))

        new_tracked_objects = {}
        
        for cx, cy, x, y, w, h in current_centroids:
            matched_id = None
            min_dist = float('inf')

            for obj_id, data in tracked_objects.items():
                old_cx, old_cy = data['centroid']
                dist = math.hypot(cx - old_cx, cy - old_cy)
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = obj_id

            sd = get_signed_distance(p1, p2, (cx, cy))

            if matched_id is not None:
                old_state = tracked_objects[matched_id]['state']
                old_cx, old_cy = tracked_objects[matched_id]['centroid']
                
                new_state = old_state
                
                if sd > crossing_margin and cv2.pointPolygonTest(paralellogram_minus, (int(old_cx), int(old_cy)), measureDist=False) >= 0:
                    new_state = 1
                elif sd < -crossing_margin and cv2.pointPolygonTest(paralellogram_plus, (int(old_cx), int(old_cy)), measureDist=False) >= 0:
                    new_state = -1

                if old_state != 0 and new_state != 0 and old_state != new_state:
                    crossed_count += 1
                    print(f"Obiekt {matched_id} przekroczył linię! (Suma: {crossed_count})")
                
                new_tracked_objects[matched_id] = {'centroid': (cx, cy), 'state': new_state}
                del tracked_objects[matched_id]
            else:
                initial_state = 0
                if sd > crossing_margin:
                    initial_state = 1
                elif sd < -crossing_margin:
                    initial_state = -1
                    
                new_tracked_objects[next_obj_id] = {'centroid': (cx, cy), 'state': initial_state}
                next_obj_id += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(frame, f"ID: {matched_id}",(x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        tracked_objects = new_tracked_objects

        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.line(frame, m1_p1, m1_p2, (0, 150, 255), 1)
        cv2.line(frame, m2_p1, m2_p2, (0, 150, 255), 1)

        cv2.putText(frame, f"Przekroczenia: {crossed_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Video", frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()