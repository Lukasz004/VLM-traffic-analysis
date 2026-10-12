"""Główny obszar odtwarzacza wideo z inferencją YOLO."""
import os
import queue
import threading
import time
from collections import defaultdict
from tkinter import filedialog

import customtkinter as ctk
import cv2
from PIL import Image
from ultralytics import YOLO

import config


class PlayerArea(ctk.CTkFrame):
    """Główny obszar odtwarzacza wideo ze statycznym rozmiarem klatki."""

    def __init__(self, parent, matrix, yolo_version):
        super().__init__(parent)

        self.current_frame = None
        self.video_writer = None
        self.is_recording = False

        self.video_source_path = None
        self.is_stream = False
        self.matrix_loaded = False
        self.is_paused = True
        self.cap = None
        self.fps = 0

        self.matrix = matrix

        self.yolo_version = yolo_version
        self._set_yolo_model()

        # self.tracks jest zapisywane w wątku _video_worker i odczytywane/
        # czyszczone z głównego wątku GUI - wymaga synchronizacji.
        self.tracks_lock = threading.Lock()
        self.erase_tracks()

        # self.cap jest tworzone/zwalniane/zerowane z głównego wątku GUI
        # (np. przy "Zmień źródło"), a jednocześnie czytane w _video_worker.
        # Bez blokady worker może dostać None w trakcie iteracji -> AttributeError.
        self.cap_lock = threading.Lock()

        self.display_queue = queue.Queue(maxsize=3)
        self.worker_running = True
        self.seek_requested = False
        self.target_frame = 0
        self.video_ended = False

        # Stan funkcji "Uruchom selekcję" - automatyczne nagrywanie fragmentów,
        # w których w strefie interakcji nachodzą się obiekty z obu grup.
        self.selection_active = False
        self.selection_out = None
        self.selection_frames_left = 0
        self.selection_recording_id = 0

        self.worker_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.worker_thread.start()

        self._build_initial_view()

    def _video_worker(self):
        while self.worker_running:
            # Pobieramy JEDNĄ referencję do cap na całą iterację - dalej w pętli
            # operujemy na zmiennej lokalnej `cap`, a nie na self.cap, żeby
            # główny wątek nie mógł "podmienić nam" obiektu na None w połowie
            # przetwarzania klatki (co powodowało AttributeError).
            with self.cap_lock:
                cap = self.cap

            if cap is None or not cap.isOpened():
                time.sleep(0.03)
                continue

            if self.seek_requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
                if not self.matrix_loaded:
                    self.matrix.reset_heatmap()
                self.seek_requested = False
                self.erase_tracks()

                # Jeśli spauzowane, załaduj podgląd 1 klatki po przesunięciu suwaka
                if self.is_paused:
                    ret, frame = cap.read()
                    if ret:
                        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if self.display_queue.full():
                            try: self.display_queue.get_nowait()
                            except queue.Empty: pass
                        self.display_queue.put((frame, frame_idx))
                continue

            if not self.is_paused:
                ret, frame = cap.read()
                if ret:
                    result = None
                    if not self.matrix_loaded or self.selection_active:
                        result = self._run_yolo_inference(frame)
                    if self.selection_active:
                        self._check_selection_zone(frame, result)
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    if self.display_queue.full():
                        try: self.display_queue.get_nowait()
                        except queue.Empty: pass
                    self.display_queue.put((frame, frame_idx))

                    delay = 1.0 / self.fps if self.fps > 0 else 0.033
                    time.sleep(delay)
                else:
                    self.video_ended = True
                    time.sleep(0.03)
            else:
                time.sleep(0.03)

    def _build_initial_view(self):
        self._clear_area()

        self.start_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.start_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Przycisk "Wybierz plik"
        self.btn_choose_file = ctk.CTkButton(
            self.start_frame, text="Wybierz plik", width=180, command=self._select_file
        )
        self.btn_choose_file.pack(pady=10)

        self.lbl_or = ctk.CTkLabel(self.start_frame, text="lub", font=ctk.CTkFont(size=12))
        self.lbl_or.pack(pady=2)

        # Pole na link strumienia
        self.entry_stream = ctk.CTkEntry(
            self.start_frame, placeholder_text="Wklej link do strumienia", width=260
        )
        self.entry_stream.pack(pady=10)
        self.entry_stream.bind("<Return>", lambda event: self._start_player())

    def _select_file(self):
        filepath = filedialog.askopenfilename(
            title="Wybierz nagranie",
            filetypes=[("Pliki wideo", "*.mp4 *.avi *.mkv *.mov")],
        )
        if filepath:
            self.video_source_path = filepath
            self.is_stream = False
            self._start_player()

    def _start_player(self):
        stream_url = self.entry_stream.get().strip()

        if stream_url:
            self.is_stream = True
            self.video_source_path = stream_url

        if self.video_source_path:
            self._build_active_view()

    def _build_active_view(self):
        self._clear_area()
        self.is_paused = False

        # Płótno wideo
        self.video_screen = ctk.CTkFrame(self, fg_color="black")
        self.video_screen.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 5))

        self.video_label = ctk.CTkLabel(self.video_screen, text="")
        self.video_label.pack(fill="both", expand=True)

        # Pasek kontrolny
        self.controls_frame = ctk.CTkFrame(self, height=50)
        self.controls_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        self.btn_play_pause = ctk.CTkButton(
            self.controls_frame, text="Zatrzymaj", width=90, command=self._toggle_play_pause
        )
        self.btn_play_pause.pack(side="left", padx=10, pady=10)

        # Otworzenie wideo cv2
        with self.cap_lock:
            self.cap = cv2.VideoCapture(self.video_source_path)
            cap = self.cap
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.matrix.set_window_size(0)

        # Suwak sekwencji widoczny TYLKO przy odtwarzaniu pliku
        if not self.is_stream and total_frames > 0:
            self.slider_seek = ctk.CTkSlider(
                self.controls_frame,
                from_=0,
                to=total_frames,
                command=self._on_seek
            )
            self.slider_seek.pack(side="left", fill="x", expand=True, padx=10, pady=10)
            self.slider_seek.set(0)

        btn_change_source = ctk.CTkButton(
            self.controls_frame,
            text="Zmień źródło",
            width=100,
            fg_color="gray40",
            command=self._reset_source,
        )
        btn_change_source.pack(side="right", padx=10, pady=10)

        # Uruchomienie pętli
        self.update_video()

    def _on_seek(self, value):
        """Obsługa przesuwania suwaka wideo."""
        with self.cap_lock:
            cap = self.cap
        if cap is not None and cap.isOpened():
            self.target_frame = int(value)
            self.seek_requested = True

    def _display_frame(self, frame):
        """Statyczna funkcja wyświetlania klatki o wymiarach 640x480."""
        # generateHeatmap samo bezpiecznie obsługuje przypadek braku heatmapy
        # (pod blokadą), więc nie sprawdzamy tu matrix_object bez synchronizacji.
        self.current_frame = self.matrix.generateHeatmap(frame)

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(self.current_frame)

        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Statyczny rozmiar (640, 480)
        ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(640, 480)
        )
        self.video_label.configure(image=ctk_image)
        self.video_label.image = ctk_image

    def update_video(self):
        # Jeśli widok został zniszczony (np. po kliknięciu "Zmień źródło"),
        # video_label już nie istnieje - kończymy tu i NIE planujemy
        # kolejnego tick, co po prostu naturalnie zatrzymuje pętlę.
        if not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
            return

        try:
            frame, frame_idx = self.display_queue.get_nowait()
            self._display_frame(frame)
            if not self.is_stream and hasattr(self, 'slider_seek') and not self.seek_requested:
                self.slider_seek.set(frame_idx)
        except queue.Empty:
            pass

        if self.video_ended:
            self.is_paused = True
            self.btn_play_pause.configure(text="Odtwórz")
            self.video_ended = False

        self.after(33, self.update_video)

    def _set_yolo_model(self):
        if self.yolo_version == "N":
            self.model = YOLO("models/yolo26n.onnx")
        elif self.yolo_version == "S":
            self.model = YOLO("models/yolo26s.onnx")
        elif self.yolo_version == "M":
            self.model = YOLO("models/yolo26m.onnx")
        elif self.yolo_version == "L":
            self.model = YOLO("models/yolo26l.onnx")
        elif self.yolo_version == "X":
            self.model = YOLO("models/yolo26x.onnx")

    def _run_yolo_inference(self, frame):
        """Uruchamia detekcję/tracking YOLO na klatce i zwraca wynik.
        Detekcje są dopisywane do self.tracks (do wyliczenia macierzy) tylko
        wtedy, gdy nie mamy wczytanej gotowej macierzy - wynik jest jednak
        zwracany zawsze, żeby mógł też zasilić funkcję "Uruchom selekcję"."""

        #persist=True jest kluczowe - nakazuje modelowi zapamiętywać obiekty i utrzymywać ich ID między klatkami
        result = self.model.track(frame, persist=True, verbose=False)[0]

        #przetwarzamy obiekty tylko, gdy tracker faktycznie przypisał im track_id
        if not self.matrix_loaded and result.boxes and result.boxes.is_track:
            track_ids = result.boxes.id.int().cpu().tolist()

            with self.tracks_lock:
                for box, track_id in zip(result.boxes, track_ids):
                    # .cpu().tolist() i jawne rzutowania (float, int) uwalniają dane z tensorów PyTorcha,
                    # co jest niezbędne, aby biblioteka json mogła je poprawnie zserializować
                    x, y, w, h = box.xywh[0].cpu().tolist()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    detection = {
                        "bbox": [x, y, w, h],
                        "confidence": conf,
                        "class_name": class_name,
                        "class_id": class_id
                    }
                    self.tracks[track_id].append(detection)

        return result

    def _check_selection_zone(self, frame, result):
        """Odpowiednik logiki z samodzielnego skryptu selekcji: sprawdza, czy
        w strefie interakcji (wczytana macierz) nachodzą się jednocześnie
        obiekty z grupy pojazdów i z grupy ludzi, i jeśli tak - nagrywa
        fragment do osobnego pliku (z buforem klatek po zaniku zdarzenia)."""
        if result is None or result.boxes is None:
            return

        matrix_object = self.matrix.get_matrix_object()
        if matrix_object is None:
            return

        height, width = frame.shape[:2]

        # Wczytana macierz mogła zostać wygenerowana dla nagrania o innej
        # rozdzielczości niż to obecnie odtwarzane - bez dopasowania rozmiaru
        # indeksowanie po współrzędnych bboxa (liczonych w rozdzielczości
        # bieżącej klatki) wypadałoby poza zakres tablicy.
        if matrix_object.shape[:2] != (height, width):
            matrix_object = cv2.resize(matrix_object, (width, height))

        trigger_group1 = False
        trigger_group2 = False

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            #zabezpieczenie przed wyjściem bboxa poza kadr
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)

            #sprawdzenie, czy którykolwiek róg bboxa nachodzi strefę interakcji
            if (matrix_object[y1, x1] > config.SELECTION_THRESHOLD or
                    matrix_object[y2, x1] > config.SELECTION_THRESHOLD or
                    matrix_object[y1, x2] > config.SELECTION_THRESHOLD or
                    matrix_object[y2, x2] > config.SELECTION_THRESHOLD):

                if class_name in config.CLASSES_VEHICLES:
                    trigger_group1 = True
                elif class_name in config.CLASSES_PEOPLE:
                    trigger_group2 = True

        #inicjalizacja zapisu, jeśli wykryto obiekty z obu grup w strefie
        if trigger_group1 and trigger_group2:
            self.selection_frames_left = config.SELECTION_BUFFER_FRAMES
            if self.selection_out is None:
                os.makedirs(config.SELECTION_OUTPUT_DIR, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = self.fps if self.fps > 0 else 30.0
                path = f"{config.SELECTION_OUTPUT_DIR}/selekcja_{self.selection_recording_id}.mp4"
                self.selection_out = cv2.VideoWriter(path, fourcc, fps, (width, height))

        #logika podtrzymania nagrywania (bufor klatek po zaniku zdarzenia)
        if self.selection_frames_left > 0 and self.selection_out is not None:
            self.selection_out.write(frame)

            if not trigger_group2:
                self.selection_frames_left -= 1

            if self.selection_frames_left == 0:
                self.selection_out.release()
                self.selection_out = None
                self.selection_recording_id += 1

    def erase_tracks(self):
        with self.tracks_lock:
            self.tracks = defaultdict(list)

    def get_tracks(self):
        with self.tracks_lock:
            return self.tracks

    def pop_tracks(self):
        """Atomowo pobiera bieżące detekcje i czyści bufor.
        Zastępuje niebezpieczną sekwencję get_tracks() + erase_tracks(),
        między którymi wątek roboczy mógł dopisać nowe detekcje."""
        with self.tracks_lock:
            tracks = self.tracks
            self.tracks = defaultdict(list)
            return tracks

    def _toggle_play_pause(self):
        """Przełączanie stanu pauzy/odtwarzania."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_play_pause.configure(text="Odtwórz")
        else:
            self.btn_play_pause.configure(text="Zatrzymaj")

    def _reset_source(self):
        self.video_source_path = None
        self.is_stream = False
        self.is_paused = True
        self._build_initial_view()

    def _clear_area(self):
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        while not self.display_queue.empty():
            try: self.display_queue.get_nowait()
            except queue.Empty: pass

        for child in self.winfo_children():
            child.destroy()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False

        if self.selection_out is not None:
            self.selection_out.release()
            self.selection_out = None
        self.selection_frames_left = 0

    def set_yolo_version(self, version):
        self.yolo_version = version
        self._set_yolo_model()

    def get_fps(self):
        return self.fps

    def check_if_playing(self):
        return not self.is_paused

    def get_video_size(self):
        with self.cap_lock:
            cap = self.cap
        if cap is None:
            return (0, 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def set_selection_active(self, active):
        """Włącza/wyłącza funkcję "Uruchom selekcję". Przy wyłączeniu
        domykamy ewentualnie trwające nagranie zdarzenia."""
        self.selection_active = active
        if not active and self.selection_out is not None:
            self.selection_out.release()
            self.selection_out = None
            self.selection_frames_left = 0

    def mark_matrix_state(self, loaded):
        self.matrix_loaded = loaded

    def save_current_frame(self, path):
        """Zapisuje bieżącą klatkę do pliku graficznego."""
        if self.current_frame is not None and path:
            cv2.imwrite(path, self.current_frame)

    def start_recording(self, path):
        """Uruchamia nagrywanie wideo do wskazanego pliku."""
        if not path:
            return
        width, height = self.get_video_size()
        if width == 0 or height == 0:
            return
        fps = self.fps if self.fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        self.is_recording = True

    def stop_recording(self):
        """Zatrzymuje nagrywanie i zamyka plik wideo."""
        self.is_recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None