import os

# Ścieżka do folderu ze zdjęciami
folder_path = r'C:\Users\tomek\Documents\MAGISTERSKIE\PRACA MAGISTERSKA\FINAL BOSS\PenaltyMaster\all_data\center'



# Nowa baza nazwy pliku
base_name = 'center'

# Numer startowy
start_number = 1000

# Pobierz wszystkie pliki obrazów
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Przejdź przez pliki i zmień nazwy
for idx, filename in enumerate(image_files):
    # Ścieżka do starego pliku
    old_path = os.path.join(folder_path, filename)

    # Nowa nazwa pliku
    extension = os.path.splitext(filename)[1]  # zachowaj rozszerzenie (.jpg, .png itd.)
    new_filename = f"{base_name}{start_number + idx}{extension}"

    # Ścieżka do nowego pliku
    new_path = os.path.join(folder_path, new_filename)

    # Zmień nazwę
    os.rename(old_path, new_path)
    print(f"Zmieniono: {filename} -> {new_filename}")

print("ZAKOŃCZONO!")
