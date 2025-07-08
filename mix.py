import os
import random

# Ustawienia
#folder_path = r'C:\Users\tomek\Documents\MAGISTERSKIE\PRACA MAGISTERSKA\FINAL BOSS\PenaltyMaster\penalty_master\data\train\left_cropped'  # <- podaj ścieżkę do folderu

import os
import random
import shutil

# === USTAWIENIA ===
folder_path = r'C:\Users\tomek\Documents\MAGISTERSKIE\PRACA MAGISTERSKA\FINAL BOSS\PenaltyMaster\penalty_master\data\train\right_cropped'
prefix = 'right'  # 'left', 'right', 'center' itd.

# Ścieżka do folderu docelowego
output_folder = os.path.join(os.path.dirname(folder_path), f"{prefix}_mix")

# Utwórz folder docelowy, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# Pobierz i przetasuj listę plików
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# Przetwarzanie i kopiowanie plików
for idx, filename in enumerate(image_files):
    old_path = os.path.join(folder_path, filename)
    extension = os.path.splitext(filename)[1]
    new_filename = f"{prefix}{idx}{extension}"
    new_path = os.path.join(output_folder, new_filename)

    shutil.copy2(old_path, new_path)  # kopiuj z metadanymi
    print(f"Skopiowano: {filename} -> {new_filename}")

print("ZAKOŃCZONO!")
