"""Punkt wejścia aplikacji."""
from config import apply_theme
from app import App


if __name__ == "__main__":
    apply_theme()
    app = App()
    app.mainloop()