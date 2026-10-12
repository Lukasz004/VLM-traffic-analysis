"""Górny pasek nawigacyjny aplikacji."""
import customtkinter as ctk
from tkinter import filedialog

from .modals import YoloWindow


class TopBar(ctk.CTkFrame):
    """Górny pasek nawigacyjny z przyciskami."""

    def __init__(self, parent, app, matrix):
        super().__init__(parent, height=45, corner_radius=0)
        self.app = app

        # Przycisk "Zapisz" z menu rozwijanym
        self.btn_save = ctk.CTkButton(
            self, text="Zapisz", width=100, command=self._toggle_save_menu
        )
        self.btn_save.pack(side="left", padx=10, pady=5)

        self.save_menu = ctk.CTkFrame(
            parent, width=120, fg_color=("gray85", "gray20"), border_width=1
        )
        self.is_recording = False
        self.btn_save_video = ctk.CTkButton(
            self.save_menu,
            text="Nagranie",
            fg_color="transparent",
            anchor="w",
            command=self._toggle_recording,
        )
        btn_save_frame = ctk.CTkButton(
            self.save_menu,
            text="Klatkę",
            fg_color="transparent",
            anchor="w",
            command=lambda: self._save_action("Klatkę"),
        )
        self.btn_save_video.pack(fill="x", padx=2, pady=2)
        btn_save_frame.pack(fill="x", padx=2, pady=2)

        # Przycisk "Wczytaj macierz"
        self.btn_load_matrix = ctk.CTkButton(
            self, text="Wczytaj macierz", width=120, command=self._load_matrix
        )
        self.btn_load_matrix.pack(side="left", padx=10, pady=5)
        self.matrixLoaded = False

        # Przycisk "Wybierz YOLO"
        self.btn_yolo = ctk.CTkButton(
            self, text="Wybierz YOLO", width=120, command=self._open_yolo_window
        )
        self.btn_yolo.pack(side="left", padx=10, pady=5)

        self.matrix = matrix

    def _toggle_recording(self):
        self.save_menu.place_forget()
        if not self.is_recording:
            path = filedialog.asksaveasfilename(
                title="Zapisz nagranie",
                defaultextension=".mp4",
                filetypes=[("Pliki MP4", "*.mp4"), ("Wszystkie pliki", "*.*")]
            )
            if path:
                self.app.player_area.start_recording(path)
                self.is_recording = True
                self.btn_save.configure(text="Zakończ nagranie", fg_color="red")
        else:
            self.app.player_area.stop_recording()
            self.is_recording = False
            self.btn_save.configure(
                text="Zapisz", 
                fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]
            )

    def _save_action(self, option_type):
        self.save_menu.place_forget()
        if option_type == "Klatkę":
            path = filedialog.asksaveasfilename(
                title="Zapisz klatkę",
                defaultextension=".jpg",
                filetypes=[("Obrazy JPEG", "*.jpg"), ("Obrazy PNG", "*.png"), ("Wszystkie pliki", "*.*")]
            )
            if path:
                self.app.player_area.save_current_frame(path)

    def _toggle_save_menu(self):
        # Jeśli trwa nagrywanie, kliknięcie przycisku bezpośrednio kończy nagrywanie
        if self.is_recording:
            self._toggle_recording()
            return

        if self.save_menu.winfo_viewable():
            self.save_menu.place_forget()
        else:
            x = self.btn_save.winfo_rootx() - self.master.winfo_rootx()
            y = (
                self.btn_save.winfo_rooty()
                - self.master.winfo_rooty()
                + self.btn_save.winfo_height()
                + 2
            )
            self.save_menu.place(x=x, y=y)
            self.save_menu.lift()

    def _load_matrix(self):
        if not self.matrixLoaded:
            path = filedialog.askopenfilename(title="Wczytaj macierz", filetypes=(("Macierz Numpy", "*.npy"),))
            if path:
                self.matrixLoaded = True
                self.app.mark_matrix_state(True)
                self.matrix.set_matrix_path(path)
                self.btn_load_matrix.configure(text="Wyczyść macierz", fg_color="red")
        else:
            self.matrixLoaded = False
            self.app.mark_matrix_state(False)
            self.matrix.set_matrix_path(None)
            self.btn_load_matrix.configure(text="Wczytaj macierz", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

    def _open_yolo_window(self):
        YoloWindow(
            parent=self.master,
            versions=self.app.yolo_versions,
            current_version=self.app.selected_yolo_version,
            on_select_callback=self.app.set_yolo_version,
        )
