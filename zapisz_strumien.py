import cv2

def record_stream(stream_url, output_filename="nagranie.mp4"):
    # 1. Otwarcie strumienia wideo
    print(f"Próba połączenia ze strumieniem: {stream_url}...")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć strumienia wideo. Sprawdź poprawność linku.")
        return

    # 2. Pobranie parametrów strumienia (szerokość, wysokość, klatki na sekundę)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Strumienie sieciowe często nie podają poprawnego FPS, ustalamy domyślne, jeśli brakuje
    if fps == 0 or fps != fps:
        print("Strumień nie podał FPS, ustawiam domyślne: 25.0")
        fps = 25.0

    # 3. Konfiguracja kodeka i obiektu do zapisu (mp4v to standardowy kodek dla MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"Pomyślnie połączono. Rozpoczęto zapis do pliku: {output_filename}")
    print("Naciśnij 'q' w oknie podglądu lub użyj Ctrl+C w terminalu, aby zakończyć nagrywanie.")

    try:
        while True:
            # Odczytywanie kolejnych klatek
            ret, frame = cap.read()
            
            if not ret:
                print("\nStrumień został przerwany lub zakończony.")
                break

            # Zapis klatki do pliku
            out.write(frame)

            # Opcjonalne: Wyświetlanie podglądu na żywo
            #cv2.imshow('Podgląd strumienia (Wciśnij Q aby wyjść)', frame)
            
            # Zakończenie po wciśnięciu klawisza 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nPrzerwano przez użytkownika (klawisz 'q').")
                break

    except KeyboardInterrupt:
        print("\nPrzerwano z poziomu konsoli (Ctrl+C).")
        
    finally:
        # 4. Zwolnienie zasobów - zamykanie plików i okien
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Zakończono pracę skryptu. Zapisano plik.")

# ==========================================
# UŻYCIE SKRYPTU
# ==========================================
if __name__ == "__main__":
    # Tutaj wklej link do swojego strumienia wideo (np. RTSP, http)
    LINK_DO_STRUMIENIA = "https://kam1.mailfull.pl/15_1_ak-niepodleglosci/memfs/97590dc3-ca4d-48b9-a2e4-efcbc0bf8260.m3u8" 
    
    # Nazwa pliku wyjściowego
    PLIK_WYNIKOWY = "moj_strumien.mp4"
    
    record_stream(LINK_DO_STRUMIENIA, PLIK_WYNIKOWY)