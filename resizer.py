import cv2
import os

folder = 'left'

# Ścieżki bazujące na zmiennej folder
input_path = f'data/train/{folder}'
output_path = f'data/train/{folder}'

image_names = os.listdir(input_path)
output_image_num = 0

for image in image_names:
    print(image)

    # Pełna ścieżka do starego obrazu
    image_path = os.path.join(input_path, image)

    # Wczytaj obraz
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Sprawdź, czy obraz został poprawnie wczytany
    if img is None:
        print(f"Problem with {image}, skip...")
        continue

    # Zmień rozdzielczość
    newimg = cv2.resize(img, (1280, 610))

    # Nowa nazwa pliku: right0.jpg, left0.jpg, center0.jpg itd.
    new_image_name = f"{folder}{output_image_num}.jpg"
    new_image_path = os.path.join(output_path, new_image_name)

    # Zapisz nowy obraz
    cv2.imwrite(new_image_path, newimg)

    # Usuń stary obraz
    os.remove(image_path)

    output_image_num += 1

print("FINISHED")
