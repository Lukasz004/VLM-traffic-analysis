"""Logika obliczeniowa macierzy gęstości / heatmapy.

Ten moduł celowo NIE zależy od Tkintera / customtkinter - operuje wyłącznie
na danych (pandas/numpy/cv2), dzięki czemu można go łatwo testować osobno
od warstwy GUI.
"""
import threading

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class Matrix:
    def __init__(self):
        super().__init__()

        self.detections = dict()
        self.window_size = 0
        self.sigma = 0
        self.displacement = 0
        self.actual_detections_pos = 0

        self.matrix_object = None
        self.dataframe = None
        self.cached_heatmap = None
        self.density = None

        self.people_classes = []
        self.vehicle_classes = []

        # matrix_object/cached_heatmap są zapisywane w wątku przetwarzającym
        # macierz (App) oraz w wątku odtwarzacza (PlayerArea przy seeku),
        # a odczytywane w głównym wątku GUI przy każdej wyświetlanej klatce.
        self.state_lock = threading.Lock()

    def set_people_classes(self, classes):
        self.people_classes = classes

    def set_vehicle_classes(self, classes):
        self.vehicle_classes = classes

    def set_window_size(self, size):
        if size != self.window_size:
            self.detections = dict()
            self.window_size = size
            self.actual_detections_pos = 0
            with self.state_lock:
                self.matrix_object = None
                self.cached_heatmap = None
            print(size)

    def get_matrix_object(self):
        """Zwraca (w sposób bezpieczny wątkowo) aktualnie wczytaną/wyliczoną
        macierz, żeby np. PlayerArea mogło sprawdzić strefę interakcji na
        potrzeby funkcji "Uruchom selekcję"."""
        with self.state_lock:
            return self.matrix_object

    def reset_heatmap(self):
        """Atomowo czyści bieżącą macierz/heatmapę (np. przy przewijaniu wideo)."""
        with self.state_lock:
            self.matrix_object = None
            self.cached_heatmap = None

    def set_displacement(self, displacement):
        self.displacement = displacement

    def add_detections(self, detections):
        self.detections[self.actual_detections_pos] = detections
        self.actual_detections_pos += 1
        if self.actual_detections_pos >= self.window_size:
            self.actual_detections_pos = 0

    def set_matrix_path(self, path):
        if path:
            matrix_object = np.load(path)
            matrix_uint8 = (np.clip(matrix_object, 0, 1) * 255).astype(np.uint8)
            cached_heatmap = cv2.applyColorMap(matrix_uint8, cv2.COLORMAP_JET)
            with self.state_lock:
                self.matrix_object = matrix_object
                self.cached_heatmap = cached_heatmap
        else:
            self.reset_heatmap()

    def flatten_dictionary(self):
        if self.window_size - 1 in self.detections:
            rows = []

            # Przechodzimy po zewnętrznych elementach (defaultdict)
            for track_dict in self.detections.values():
                # Przechodzimy po kluczach (track_id) i listach detekcji
                for track_id, detections_list in track_dict.items():
                    for det in detections_list:
                        # Rozpakowujemy listę bbox na x, y, w, h
                        x, y, w, h = det["bbox"]

                        # Tworzymy płaski słownik dla jednego wiersza
                        rows.append(
                            {
                                "track_id": track_id,
                                "class_id": det["class_id"],
                                "class_name": det["class_name"],
                                "confidence": det["confidence"],
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                            }
                        )

            # Tworzymy DataFrame z przygotowanej listy wierszy
            self.dataframe = pd.DataFrame(rows)
            self.calculate_shifts()
            self.classify_tracks()
            if self.matrix_type == "moving":
                matrix_object = self.trainMatrix(self.moving_detections)
            elif self.matrix_type == "static":
                matrix_object = self.trainMatrix(self.static_detections)
            elif self.matrix_type == "interaction":
                det1 = self.moving_detections[self.moving_detections['class_name'].isin(self.vehicle_classes)]
                det2 = self.moving_detections[self.moving_detections['class_name'].isin(self.people_classes)]

                matrix1 = self.trainMatrix(det1)
                matrix2 = self.trainMatrix(det2)

                resultMatrix = matrix1 * matrix2
                max_val = np.ndarray.max(resultMatrix)
                if max_val > 0:
                    resultMatrix = resultMatrix / max_val

                matrix_object = resultMatrix
            else:
                matrix_object = None

            if matrix_object is not None:
                # Buforowanie heatmapy - liczone lokalnie, żeby nie trzymać
                # blokady przez czas trwania obliczeń (gaussian_filter, cv2).
                matrix_uint8 = (np.clip(matrix_object, 0, 1) * 255).astype(np.uint8)
                cached_heatmap = cv2.applyColorMap(matrix_uint8, cv2.COLORMAP_JET)

                with self.state_lock:
                    self.matrix_object = matrix_object
                    self.cached_heatmap = cached_heatmap

    def calculate_shifts(self):

        self.dataframe['x_past'] = self.dataframe.groupby('track_id')['x'].shift(self.displacement)
        self.dataframe['y_past'] = self.dataframe.groupby('track_id')['y'].shift(self.displacement)

        euclidean_distance = np.sqrt(
            (self.dataframe['x'] - self.dataframe['x_past'])**2 +
            (self.dataframe['y'] - self.dataframe['y_past'])**2
        )

        self.dataframe['przesunięcie'] = euclidean_distance.fillna(0.0)
        self.dataframe['przesunięcie'] = self.dataframe['przesunięcie'].round(2)
        self.dataframe = self.dataframe.drop(columns=['x_past', 'y_past'])

    def classify_tracks(self):
        PROGI_KLAS = {
            'car': 50.0,
            'truck': 50.0,
            'bus': 50.0,
            'motorcycle': 50.0,
            'person': 10.0,
            'umbrella': 10.0
        }

        srednie_przesunięcia = self.dataframe.groupby('track_id').agg(
        srednie_przesunięcie=('przesunięcie', 'mean'),
        class_name=('class_name', 'first')
        ).reset_index()

        srednie_przesunięcia['wymagany_prog'] = srednie_przesunięcia['class_name'].map(PROGI_KLAS)

        ruch_df = srednie_przesunięcia[srednie_przesunięcia['srednie_przesunięcie'] >= srednie_przesunięcia['wymagany_prog']]
        bezruch_df = srednie_przesunięcia[srednie_przesunięcia['srednie_przesunięcie'] < srednie_przesunięcia['wymagany_prog']]

        ruch = ruch_df['track_id'].tolist()
        bezruch = bezruch_df['track_id'].tolist()

        self.moving_detections = self.dataframe[self.dataframe['track_id'].isin(ruch)]
        self.static_detections = self.dataframe[self.dataframe['track_id'].isin(bezruch)]

    def set_train_parameters(self, width, height, sigma, matrix_type):
        self.width = width
        self.height = height
        self.sigma = sigma
        self.matrix_type = matrix_type

    def trainMatrix(self, detections):
        if self.density is None or self.density.shape != (self.height, self.width):
            self.density = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            self.density.fill(0.0)

        for row in detections.itertuples():
            w = int(row.w)
            h = int(row.h)
            x1 = int(row.x - w/2)
            x2 = int(row.x + w/2)
            y1 = int(row.y - h/2)
            y2 = int(row.y + h/2)
            cx = int(row.x)
            cy = int(row.y)

            #sprawdzamy czy ramka jest co najmniej piksel od krawedzi obrazu
            if (10 <= x1 < self.width-10) and (10 <= y1 < self.height-10) and (10 <= x2 < self.width-10) and (10 <= y2 < self.height-10):
                self.density[cy, cx] += 1.0
                self.density[y1:y2, x1:x2] += 0.4

        smoothed = gaussian_filter(self.density, sigma=self.sigma)
        max_val = smoothed.max()
        if max_val > 0:
            smoothed /= max_val

        return smoothed


    def generateHeatmap(self, first_frame):
        with self.state_lock:
            heatmap = self.cached_heatmap

        if heatmap is None:
            return first_frame

        # Opcjonalnie: upewnij się, że heatmap ma ten sam rozmiar co first_frame
        if heatmap.shape[:2] != first_frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (first_frame.shape[1], first_frame.shape[0]))

        result_frame = cv2.addWeighted(first_frame, 0.5, heatmap, 0.5, 0)

        return result_frame