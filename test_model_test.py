from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import cv2
from math import ceil
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Ścieżki
model_dir = 'model'
model_name = 'modelPM.h5'
image_path = 'data/test/mix'

# Parametry
image_width, image_height = 256, 256
grid_columns = 6
stats_width = 300

# Załaduj model
model = load_model(os.path.join(model_dir, model_name))

# Obrazy i etykiety
image_names = os.listdir(image_path)
true_labels = [0 if 'center' in name else 1 if 'left' in name else 2 for name in image_names]
label_map = {0: "center", 1: "left", 2: "right"}

# Predykcje i statystyki
predictions = []
class_correct = {0: 0, 1: 0, 2: 0}
class_total = {0: 0, 1: 0, 2: 0}

grid_rows = ceil(len(image_names) / grid_columns)
grid_image = np.ones(((image_height + 50) * grid_rows, image_width * grid_columns + stats_width, 3), dtype=np.uint8) * 255

# Przetwarzanie obrazów
for idx, image_name in enumerate(image_names):
    true_label = true_labels[idx]
    img_path = os.path.join(image_path, image_name)
    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img_resized = cv2.resize(img, (image_width, image_height))
    img_normalized = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_to_array(img_normalized), axis=0)

    pred = model.predict(img_array)
    pred_label = pred.argmax(axis=1)[0]

    predictions.append(pred_label)
    class_total[true_label] += 1
    if pred_label == true_label:
        class_correct[true_label] += 1

    row = idx // grid_columns
    col = idx % grid_columns
    x_offset = col * image_width
    y_offset = row * (image_height + 50)

    grid_image[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = img_resized
    pred_text = f"Pred: {label_map[pred_label]}"
    true_text = f"True: {label_map[true_label]}"
    color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)

    cv2.putText(grid_image, pred_text, (x_offset, y_offset + image_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(grid_image, true_text, (x_offset, y_offset + image_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# Całościowa accuracy
correct_predictions = sum(class_correct.values())
overall_accuracy = correct_predictions / len(image_names) * 100
cv2.putText(grid_image, f"Overall Accuracy: {overall_accuracy:.2f}%", (image_width * grid_columns + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Accuracy dla każdej klasy
y_offset_stats = 70
for k in class_total:
    acc = (class_correct[k] / class_total[k] * 100) if class_total[k] > 0 else 0
    cv2.putText(grid_image, f"{label_map[k].capitalize()} Accuracy: {acc:.2f}%", (image_width * grid_columns + 20, y_offset_stats),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_offset_stats += 30

# === TABELA ===
def create_classification_table_image(report_dict, class_acc_dict, output_file="classification_table_temp.png"):
    classes = ["center", "left", "right"]
    metrics = ["precision", "recall", "f1-score", "accuracy"]
    data = []

    for cls in classes:
        row = [
            f"{report_dict[cls]['precision']:.2f}",
            f"{report_dict[cls]['recall']:.2f}",
            f"{report_dict[cls]['f1-score']:.2f}",
            f"{class_acc_dict[cls]:.2f}"
        ]
        data.append(row)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=metrics, rowLabels=classes, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    fig.patch.set_facecolor('white')
    plt.savefig(output_file, bbox_inches='tight', dpi=200)
    plt.close()

# Przygotowanie danych do tabeli
report = classification_report(true_labels, predictions, target_names=["center", "left", "right"], output_dict=True)
class_accuracies = {
    "center": (class_correct[0] / class_total[0] * 100) if class_total[0] > 0 else 0,
    "left": (class_correct[1] / class_total[1] * 100) if class_total[1] > 0 else 0,
    "right": (class_correct[2] / class_total[2] * 100) if class_total[2] > 0 else 0,
}
create_classification_table_image(report, class_accuracies)

# Załaduj obrazek tabeli i doklej pod siatką
table_img = cv2.imread("classification_table_temp.png")
if table_img is not None:
    table_height, table_width = table_img.shape[:2]
    final_image = np.ones((grid_image.shape[0] + table_height + 20, max(grid_image.shape[1], table_width), 3), dtype=np.uint8) * 255
    final_image[:grid_image.shape[0], :grid_image.shape[1]] = grid_image
    final_image[grid_image.shape[0] + 20:grid_image.shape[0] + 20 + table_height, :table_width] = table_img
else:
    final_image = grid_image

cv2.imwrite("final_test_results_full.png", final_image)
