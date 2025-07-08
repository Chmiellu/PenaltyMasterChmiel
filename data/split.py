import os
import shutil
import random

# Ścieżki główne
base_dir = os.path.abspath(os.path.dirname(__file__))  # zakłada, że skrypt jest w folderze 'data'
all_dir = os.path.join(base_dir, 'all')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validate')
test_dir = os.path.join(base_dir, 'test')

# Mapowanie folderów źródłowych na docelowe
category_map = {
    'center_mix': 'center',
    'left_mix': 'left',
    'right_mix': 'right',
}

# Funkcja do kopiowania zdjęć
def distribute_images(source_subfolder, dest_name):
    source_path = os.path.join(all_dir, source_subfolder)
    images = os.listdir(source_path)
    random.shuffle(images)

    total = len(images)
    n_train = int(0.80 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val  # zostaje 5%

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # Funkcja kopiująca do odpowiedniego folderu
    def copy_group(img_list, target_root):
        target_path = os.path.join(target_root, dest_name)
        os.makedirs(target_path, exist_ok=True)
        for img in img_list:
            shutil.copy(
                os.path.join(source_path, img),
                os.path.join(target_path, img)
            )

    copy_group(train_imgs, train_dir)
    copy_group(val_imgs, val_dir)
    copy_group(test_imgs, test_dir)

# Wykonaj dla każdej kategorii
for mix_folder, dest_folder in category_map.items():
    distribute_images(mix_folder, dest_folder)

print("SUCESSFULLY SPLIT DATASET!")
