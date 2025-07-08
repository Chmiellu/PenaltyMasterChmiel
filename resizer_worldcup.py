import cv2
import os

# Ścieżka do folderu wejściowego
input_path = r'C:\Users\tomek\Downloads\fifa final\woman_center'
output_path = os.path.join(os.path.dirname(input_path), 'fifa_woman_center')

# Procent przycięcia środka (np. 80 oznacza 80% środka kwadratu)
crop_percent = 45

os.makedirs(output_path, exist_ok=True)

image_names = os.listdir(input_path)
output_image_num = 0

for image in image_names:
    print(image)

    image_path = os.path.join(input_path, image)

    if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Pomijam plik: {image}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Nie mogę wczytać: {image}, pomijam...")
        continue

    h, w, _ = img.shape

    # Krok 1: przycięcie do kwadratu (środek)
    if w > h:
        x1 = (w - h) // 2
        img_cropped = img[:, x1:x1 + h]
    else:
        y1 = (h - w) // 2
        img_cropped = img[y1:y1 + w, :]

    # Krok 2: przycięcie procentowe środka
    square_size = img_cropped.shape[0]
    final_crop_size = int(square_size * crop_percent / 100)
    start = (square_size - final_crop_size) // 2
    img_final = img_cropped[start:start + final_crop_size, start:start + final_crop_size]

    # Krok 3: skalowanie do 256x256
    img_resized = cv2.resize(img_final, (256, 256))

    # Zapis obrazu
    new_image_name = f"centerwoman{output_image_num}.jpg"
    new_image_path = os.path.join(output_path, new_image_name)
    cv2.imwrite(new_image_path, img_resized)

    output_image_num += 1

print("FINISHED — Nowe kwadratowe obrazy są gotowe!")
