"""Stałe konfiguracyjne aplikacji (motyw, wersje YOLO, listy klas)."""
import customtkinter as ctk


def apply_theme():
    """Ustawia globalny motyw customtkinter. Wywołać raz, na starcie aplikacji."""
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")


# Dostępne wersje modeli YOLO
YOLO_VERSIONS = ["N", "S", "M", "L", "X"]
DEFAULT_YOLO_VERSION = "N"

# UWAGA: nazwy klas muszą dokładnie odpowiadać `model.names` z YOLO
# (dla domyślnych wag COCO to np. "motorcycle", nie "motorbike").
# Jeśli używasz modelu trenowanego na własnym zbiorze klas,
# zaktualizuj obie listy zgodnie z jego `model.names`.
CLASSES_VEHICLES = ["car", "motorcycle", "bus", "train", "truck", "bicycle"]
CLASSES_PEOPLE = ["person"]

# Parametry funkcji "Uruchom selekcję" - automatyczne wycinanie fragmentów
# nagrania, w których w strefie interakcji (wczytana macierz) jednocześnie
# pojawia się obiekt z grupy pojazdów i z grupy ludzi.
SELECTION_THRESHOLD = 0.1
SELECTION_BUFFER_FRAMES = 60
SELECTION_OUTPUT_DIR = "selections"