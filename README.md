# Analiza nagrań ruchu drogowego z wykorzystaniem modeli wizualno-językowych

Niniejsze repozytorium zawiera kody źródłowe systemu służącego do analizy nagrań z ruchu drogowego za pomocą modeli wizualno-językowych (VLM).
Repozytorium jest w trakcie rozwoju. Dotychczas system składa się z następujących etapów analizy:

1. Generacja pliku JSON z wykrytymi obiektami na nagraniu za pomocą sieci YOLO. `video_to_json.py` - skrypt uruchamiany na maszynie z GPU tworzący pliki wynikowe `film*.json`.
2. Filtracja obiektów niepożądanych klas, określenie które obiekty się poruszają, a które stoją. Notatnik `aktywacja_tracking.ipynb`
3. Generacja map gęstości i macierzy interakcji używanej potem w procesie selekcji. Notatnik `aktywacja_tracking.ipynb`
4. Selekcja fragmentów interesujących z punktu widzenia bezpieczeństwa ruchu drogowego. Aktywacja przycinania nagrania poprzez wyzwalacz będący jednoczesnym 
pojawieniem się pieszego i pojazdu w obszarze interakcji. Skrypt `selekcja.py`.
5. Analiza nagrania z wykorzystaniem modelu wizualno-językowego. Notatnik `model qwen38.ipynb`.

Projekt udostępnia również interfejs graficzny w którym można testować generowanie map gęstości na żywo z nagrania wideo lub strumienia. Pozwala ono również na
wczytanie wygenerowanej wcześniej macierzy interakcji i przeprowadzenie na jej podstawie procesu selekcji. Interfejs graficzny można uruchomić poprzez
`python main.py`.
