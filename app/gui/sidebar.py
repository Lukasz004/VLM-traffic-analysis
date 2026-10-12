"""Pasek boczny z parametrami."""
import customtkinter as ctk

from .modals import ClassSelectionWindow


class Sidebar(ctk.CTkScrollableFrame):
    """Pasek boczny umieszczony po prawej stronie."""

    def __init__(self, parent, app):
        super().__init__(parent, width=280, label_text="Parametry")
        self.app = app

        self.info = ctk.CTkLabel(self, text="Przy wczytanej macierzy nie można\n manipulować parametrami", justify="left")
        self.info.pack(side="left")

        # Uruchom selekcję
        self._add_title("Uruchom selekcję", self)
        self.cb_selection = ctk.CTkCheckBox(self, text="Aktywuj", command=self._toggle_selection)
        self.cb_selection.pack(anchor="w", pady=(0, 10))

        self.params_frame = ctk.CTkFrame(self, fg_color="transparent")

        # Długość okna czasowego
        self.window_size = 0
        self._add_title("Długość okna czasowego", self.params_frame)
        f_time = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        f_time.pack(fill="x", pady=(0, 10))
        self.entry_time_window = ctk.CTkEntry(f_time, width=120)
        self.entry_time_window.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(f_time, text="sek").pack(side="left")
        self.entry_time_window.bind("<Return>", lambda event: self._set_window_size(int(self.entry_time_window.get())))

        # Rozmycie sigma
        self.sigma = 0
        self._add_title("Rozmycie sigma", self.params_frame)
        self.entry_sigma = ctk.CTkEntry(self.params_frame, width=120)
        self.entry_sigma.insert(0, str(self.sigma))
        self.entry_sigma.pack(anchor="w", pady=(0, 10))
        self.entry_sigma.bind("<Return>", lambda event: self._set_sigma(float(self.entry_sigma.get())))

        self.checkbox_state = ctk.StringVar(value="moving")

        # Radiobuttony stanów obiektów
        self.rb_moving = ctk.CTkRadioButton(
            self.params_frame,
            text="obiekty poruszające się",
            variable=self.checkbox_state,
            value="moving",
        )
        self.rb_moving.pack(anchor="w", pady=4)

        self.rb_static = ctk.CTkRadioButton(
            self.params_frame,
            text="obiekty w bezruchu",
            variable=self.checkbox_state,
            value="static",
        )
        self.rb_static.pack(anchor="w", pady=4)

        self.rb_interaction = ctk.CTkRadioButton(
                    self.params_frame,
                    text="macierz interakcji",
                    variable=self.checkbox_state,
                    value="interaction",
                )
        self.rb_interaction.pack(anchor="w", pady=(4, 15))

        # Klasy pojazdy
        self._add_title("Klasy pojazdy", self.params_frame)
        btn_veh = ctk.CTkButton(
            self.params_frame, text="Wybierz", command=lambda: self._open_class_window("pojazdy")
        )
        btn_veh.pack(anchor="w", pady=(0, 15))

        # Klasy ludzie
        self._add_title("Klasy ludzie", self.params_frame)
        btn_ppl = ctk.CTkButton(
            self.params_frame, text="Wybierz", command=lambda: self._open_class_window("ludzie")
        )
        btn_ppl.pack(anchor="w", pady=(0, 15))

        # Przemieszczenie
        # UWAGA: wartość 0 oznacza shift(0), czyli x_past == x -> przesunięcie
        # zawsze wynosi 0, więc NIC nigdy nie zostałoby zakwalifikowane jako
        # "poruszające się". Domyślnie ustawiamy sensowną wartość > 0.
        self.displacement = 5
        self._add_title("Przemieszczenie", self.params_frame)
        f_disp = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        f_disp.pack(fill="x", pady=(0, 15))
        self.entry_displacement = ctk.CTkEntry(f_disp, width=80)
        self.entry_displacement.insert(0, str(self.displacement))
        self.entry_displacement.pack(side="left", padx=(0, 5))
        self.entry_displacement.bind("<Return>", lambda event: self._set_displacement(int(self.entry_displacement.get())))
        ctk.CTkLabel(f_disp, text="klatek wstecz").pack(side="left")

        self.show_parameters()

    def show_parameters(self):
        self.params_frame.pack(anchor="w", pady=(0, 10))
        self.info.pack_forget()

    def hide_parameters(self):
        self.params_frame.pack_forget()
        self.info.pack(side="left")

    def _set_window_size(self, size):
        self.window_size = size

    def get_window_size(self):
        return self.window_size

    def _set_sigma(self, sigma):
        self.sigma = sigma

    def get_sigma(self):
        return self.sigma

    def _set_displacement(self, displacement):
        self.displacement = displacement

    def get_displacement(self):
        return self.displacement

    def _add_title(self, text, location):
        lbl = ctk.CTkLabel(location, text=text, font=ctk.CTkFont(size=12, weight="bold"))
        lbl.pack(anchor="w", pady=(5, 2))

    def _open_class_window(self, category):
        is_vehicles = category == "pojazdy"
        title = "Wybierz klasy pojazdy" if is_vehicles else "Wybierz klasy ludzie"
        classes_list = self.app.classes_vehicles if is_vehicles else self.app.classes_people
        selected_dict = self.app.selected_vehicles if is_vehicles else self.app.selected_people
        callback = self.app.set_selected_vehicles if is_vehicles else self.app.set_selected_people

        ClassSelectionWindow(
            parent=self.master,
            title_text=title,
            classes_list=classes_list,
            selected_dict=selected_dict,
            on_save_callback=callback,
        )

    def check_matrix_state(self):
        return self.checkbox_state.get()

    def _toggle_selection(self):
        self.app.set_selection_active(self.cb_selection.get() == 1)