"""Główna klasa aplikacji łącząca odseparowane moduły GUI."""
import os
import threading

import customtkinter as ctk
from ultralytics import YOLO

import config
from matrix import Matrix
from gui.top_bar import TopBar
from gui.sidebar import Sidebar
from gui.player_area import PlayerArea


class App(ctk.CTk):
    """Główne okno aplikacji łączące odseparowane moduły."""

    def __init__(self):
        super().__init__()

        self.title("Aplikacja do analizy nagrań")
        self.geometry("1100x700")
        self.minsize(900, 600)

        # Dane konfiguracji aplikacji
        self.selected_yolo_version = config.DEFAULT_YOLO_VERSION
        self.yolo_versions = config.YOLO_VERSIONS

        self.classes_vehicles = config.CLASSES_VEHICLES
        self.classes_people = config.CLASSES_PEOPLE

        self.selected_vehicles = {cls: False for cls in self.classes_vehicles}
        self.selected_people = {cls: False for cls in self.classes_people}

        # Układ siatki
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)  # Lewa strona: Odtwarzacz
        self.grid_columnconfigure(1, weight=0)  # Prawa strona: Sidebar

        # Inicjalizacja poszczególnych komponentów
        self.matrix = Matrix()

        self.top_bar = TopBar(parent=self, app=self, matrix=self.matrix)
        self.top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))

        self.player_area = PlayerArea(parent=self, matrix=self.matrix, yolo_version=self.selected_yolo_version)
        self.player_area.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.sidebar = Sidebar(parent=self, app=self)
        self.sidebar.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Flaga chroniąca przed nałożeniem się dwóch cykli przetwarzania
        # macierzy, gdyby poprzedni jeszcze się nie zakończył.
        self.matrix_busy = False
        self.matrix_loaded = False

        self.after(200, self.prepare_yolo_models)
        self.update_matrix()

    def prepare_yolo_models(self):
        """Wyświetla okno ładowania i przygotowuje (eksportuje) modele YOLO w tle."""
        # 1. Tworzymy okno popup (modalne) nad głównym oknem
        popup = ctk.CTkToplevel(self)
        popup.title("Przygotowywanie modeli YOLO")
        popup.geometry("400x160")
        popup.resizable(False, False)

        # Trzymamy okno zawsze na wierzchu i blokujemy interakcję z oknem głównym
        popup.transient(self)
        popup.grab_set()

        lbl = ctk.CTkLabel(
            popup,
            text="Pobieranie i eksport modeli YOLO...Proszę czekać.",
        )
        lbl.pack(pady=(20, 10))

        progress = ctk.CTkProgressBar(popup, width=300, mode="indeterminate")
        progress.pack(pady=10)
        progress.start()

        # 2. Funkcja wykonywana w tle
        def export_worker():
            for ver in self.yolo_versions:
                model_name = f"yolo26{ver.lower()}.pt"
                output_dir = "models"

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Pobieramy/eksportujemy tylko jeśli model onnx jeszcze nie istnieje
                if not os.path.exists(f"{output_dir}/yolo26{ver.lower()}.onnx"):
                    lbl.configure(text=f"Przygotowywanie modelu: {model_name}...")
                    os.chdir(output_dir)
                    model = YOLO(model_name)
                    model.export(format="onnx")
                    os.chdir("..")

            # Po zakończeniu zamykamy popup w głównym wątku GUI.
            # UWAGA: CTkToplevel na Windows planuje wewnętrzny callback
            # (self.after(...)) do poprawnego ustawienia koloru paska tytułu
            # tuż po utworzeniu okna. Jeśli export_worker skończy się od razu
            # (np. modele .onnx już istnieją) i zniszczymy popup natychmiast,
            # ten wewnętrzny callback odpali się PO destroy() i Tkinter rzuci
            # "bad window path name .!ctktoplevel". Dajemy więc oknu minimalną
            # żywotność i dodatkowo łapiemy błąd na wszelki wypadek.
            self.after(0, lambda: self._close_prepare_popup(popup))

        # 3. Uruchomienie wątku roboczego
        threading.Thread(target=export_worker, daemon=True).start()

    def _close_prepare_popup(self, popup):
        """Bezpiecznie zamyka popup przygotowania modeli YOLO."""
        def _destroy():
            try:
                popup.grab_release()
            except Exception:
                pass
            try:
                popup.destroy()
            except Exception:
                pass

        # Minimalny odstęp, żeby wewnętrzny callback CTkToplevel zdążył
        # się wykonać przed zniszczeniem okna.
        popup.after(300, _destroy)

    def update_matrix(self):
        """Wywoływane co 1s przez self.after - działa na wątku GUI, więc
        odczytuje TYLKO lekkie wartości z widgetów Tkintera (co jest
        bezpieczne tylko na wątku głównym), a właściwe, ciężkie obliczenia
        (pandas/gaussian_filter/cv2) zleca do wątku roboczego, żeby nie
        zamrażać interfejsu."""
        if self.player_area.check_if_playing() and not self.matrix_busy and not self.matrix_loaded:
            ws = self.sidebar.get_window_size()
            sigma = self.sidebar.get_sigma()
            displacement = self.sidebar.get_displacement()
            check_state = self.sidebar.check_matrix_state()
            width, height = self.player_area.get_video_size()

            # Atomowe pobranie i wyczyszczenie zebranych detekcji.
            tracks = self.player_area.pop_tracks()

            self.matrix_busy = True
            threading.Thread(
                target=self._process_matrix,
                args=(ws, sigma, displacement, tracks, check_state, width, height),
                daemon=True,
            ).start()

        self.after(1000, self.update_matrix)

    def _process_matrix(self, ws, sigma, displacement, tracks, check_state, width, height):
        """Ciężka część przetwarzania macierzy, wykonywana w osobnym wątku.
        NIE wolno tu odwoływać się do widgetów Tkintera - Tk/Tcl nie jest
        bezpieczny wielowątkowo, dlatego wszystkie potrzebne wartości są
        przekazywane jako argumenty, odczytane wcześniej na wątku GUI."""
        try:
            self.matrix.set_window_size(ws)
            self.matrix.set_displacement(displacement)
            self.matrix.add_detections(tracks)
            self.matrix.flatten_dictionary()

            if check_state in ("moving", "static", "interaction"):
                self.matrix.set_train_parameters(width, height, sigma=sigma, matrix_type=check_state)
        except Exception as e:
            print(f"Błąd podczas przetwarzania macierzy: {e}")
        finally:
            self.matrix_busy = False

    # Metody dostępowe dla komponentów
    def set_yolo_version(self, version):
        self.selected_yolo_version = version
        self.player_area.set_yolo_version(self.selected_yolo_version)

    def set_selected_vehicles(self, selected_dict):
        self.selected_vehicles = selected_dict
        list_selected = [cls for cls, selected in selected_dict.items() if selected]
        self.matrix.set_vehicle_classes(list_selected)

    def set_selected_people(self, selected_dict):
        self.selected_people = selected_dict
        list_selected = [cls for cls, selected in selected_dict.items() if selected]
        self.matrix.set_people_classes(list_selected)

    def set_selection_active(self, active):
        self.player_area.set_selection_active(active)

    def mark_matrix_state(self, loaded):
        self.matrix_loaded = loaded
        self.player_area.mark_matrix_state(loaded)
        if loaded:
            self.sidebar.hide_parameters()
        else:
            self.sidebar.show_parameters()