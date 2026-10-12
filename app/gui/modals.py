"""Okna modalne (Toplevel) aplikacji."""
import customtkinter as ctk


class YoloWindow(ctk.CTkToplevel):
    """Okno modalne do wyboru wersji YOLO."""

    def __init__(self, parent, versions, current_version, on_select_callback):
        super().__init__(parent)
        self.title("Wybierz wersję YOLO")
        self.geometry("300x250")
        self.transient(parent)
        self.grab_set()

        self.on_select_callback = on_select_callback
        self.selected_var = ctk.StringVar(value=current_version)

        self.matrix_path = ""

        lbl = ctk.CTkLabel(
            self,
            text="Wybierz wersję YOLO",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        lbl.pack(pady=10)

        for ver in versions:
            rb = ctk.CTkRadioButton(
                self, text=ver, value=ver, variable=self.selected_var
            )
            rb.pack(anchor="w", padx=40, pady=4)

        btn_ok = ctk.CTkButton(self, text="OK", width=80, command=self._confirm)
        btn_ok.pack(pady=15)

    def _confirm(self):
        self.on_select_callback(self.selected_var.get())
        self.destroy()


class ClassSelectionWindow(ctk.CTkToplevel):
    """Okno modalne do wyboru klas (pojazdy lub ludzie)."""

    def __init__(self, parent, title_text, classes_list, selected_dict, on_save_callback):
        super().__init__(parent)
        self.title(title_text)
        self.geometry("360x300")
        self.transient(parent)
        self.grab_set()

        self.on_save_callback = on_save_callback
        self.temp_vars = {}

        lbl = ctk.CTkLabel(
            self, text=title_text, font=ctk.CTkFont(size=14, weight="bold")
        )
        lbl.pack(pady=10)

        grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Generowanie checkboxów w dwóch kolumnach
        for i, cls_name in enumerate(classes_list):
            row = i // 2
            col = i % 2
            var = ctk.BooleanVar(value=selected_dict.get(cls_name, False))
            self.temp_vars[cls_name] = var

            cb = ctk.CTkCheckBox(grid_frame, text=cls_name, variable=var)
            cb.grid(row=row, column=col, sticky="w", padx=15, pady=5)

        btn_confirm = ctk.CTkButton(self, text="Zatwierdź", command=self._confirm)
        btn_confirm.pack(pady=15)

    def _confirm(self):
        result = {cls: var.get() for cls, var in self.temp_vars.items()}
        self.on_save_callback(result)
        self.destroy()
