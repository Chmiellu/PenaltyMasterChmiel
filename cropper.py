import cv2
import os

# <<< TYLKO TĘ LINIJKĘ ZMIENIASZ >>>
folder = 'center'  # 'right', 'left', albo 'center'

# Ścieżki do folderów
input_path = f'data/train/{folder}'
output_path = f'data/train/{folder}_cropped'

# Utwórz folder wyjściowy jeśli nie istnieje
os.makedirs(output_path, exist_ok=True)

# Wczytaj pliki do obróbki
original_image_names = sorted([f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

output_image_num = 0

for image in original_image_names:
    print(f"Przetwarzanie: {image}")

    # Pełna ścieżka do obrazu
    image_path = os.path.join(input_path, image)

    # Wczytaj obraz
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Sprawdź, czy obraz został poprawnie wczytany
    if img is None:
        print(f"Problem z plikiem {image}, pomijam...")
        continue

    height, width = img.shape[:2]

    # Dynamiczne wyliczanie przycinania jako % szerokości i wysokości
    x_min = int(width * 0.30)  # np. od 30% szerokości
    x_max = int(width * 0.60)  # do 60% szerokości
    y_min = int(height * 0.20)  # od 20% wysokości
    y_max = int(height * 0.80)  # do 85% wysokości

    # Zabezpieczenie: upewnij się, że współrzędne mają sens
    if x_min >= x_max or y_min >= y_max:
        print(f"Nieprawidłowe wymiary przycinania dla pliku {image}, pomijam...")
        continue

    # Przycięcie obrazu
    crop = img[y_min:y_max, x_min:x_max]

    # Sprawdzenie, czy przycięty obraz nie jest pusty
    if crop.size == 0:
        print(f"Przycięty obraz {image} jest pusty, pomijam...")
        continue

    # --- DODANA LINIJKA: skalowanie do 256x256 ---
    crop_resized = cv2.resize(crop, (256, 256))

    # Przygotuj nową nazwę pliku
    new_image_name = f"{folder}{output_image_num}.jpg"
    new_image_path = os.path.join(output_path, new_image_name)

    # Zapisz przycięty i przeskalowany obraz
    success = cv2.imwrite(new_image_path, crop_resized)

    if not success:
        print(f"Nie udało się zapisać {new_image_name}, pomijam...")
        continue

    output_image_num += 1

print("ZAKOŃCZONO")
