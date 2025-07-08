from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import cv2
import time

# Oryginalne współrzędne
Y_MIN_OLD = 298
Y_MAX_OLD = 554
X_MIN_OLD = 450
X_MAX_OLD = 706

# Oblicz środek starego wycinka
center_y = (Y_MIN_OLD + Y_MAX_OLD) // 2
center_x = (X_MIN_OLD + X_MAX_OLD) // 2

# Nowy rozmiar 2x
height_new = (Y_MAX_OLD - Y_MIN_OLD) * 2
width_new = (X_MAX_OLD - X_MIN_OLD) * 2

# Rozmiar klatki
frame_height = 720
frame_width = 1280

# Bazowe granice nowego wycinka
Y_MIN = max(center_y - height_new // 2, 0)
Y_MAX = min(center_y + height_new // 2, frame_height)
X_MIN = max(center_x - width_new // 2, 0)
X_MAX = min(center_x + width_new // 2, frame_width)

# Dodatkowe rozszerzenie w prawo i górę
extra_right = 30
extra_top = 30

X_MAX = min(X_MAX + extra_right, frame_width)
Y_MIN = max(Y_MIN - extra_top, 0)

# Załaduj model
model_dir = 'model'
model_name = 'modelPM.h5'
model = load_model(os.path.join(model_dir, model_name))

# Wideo
video_path = 'data/video'
video_name = 'jawor.mp4'
vid = cv2.VideoCapture(os.path.join(video_path, video_name))
vid.set(cv2.CAP_PROP_POS_MSEC, 33.3)

# Przygotuj folder do zapisu klatek
output_folder = 'output_frames'
os.makedirs(output_folder, exist_ok=True)
frame_count = 0

while True:
    start = time.time()
    ret, frame = vid.read()
    if not ret:
        break

    # Wytnij i przeskaluj
    crop = frame[Y_MIN:Y_MAX, X_MIN:X_MAX]
    img = cv2.resize(crop, (256, 256))
    img = img.astype('float') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predykcja
    preds = model.predict(img)[0]
    pred_class = np.argmax(preds)

    label_map = {0: "center", 1: "right", 2: "left"}
    label = label_map.get(pred_class, "unknown")

    end = time.time()
    fps = str(int(1 / (end - start)))

    # Etykiety
    base_x = 940
    base_y = 30
    line_height = 40
    box_width = 320

    labels = [
        (f"Frame {frame_count } : {fps} FPS", (3, 30)),
        (f"Prediction: {label}", (base_x, base_y)),
        (f"Left:   {preds[2]:.2f}", (base_x, base_y + line_height)),
        (f"Center: {preds[0]:.2f}", (base_x, base_y + 2 * line_height)),
        (f"Right:  {preds[1]:.2f}", (base_x, base_y + 3 * line_height)),
    ]

    for _, (x, y) in labels[1:]:
        cv2.rectangle(frame, (x - 10, y - 25), (x - 10 + box_width, y + 5), (255, 255, 255), -1)
    cv2.rectangle(frame, (0, 5), (320, 40), (255, 255, 255), -1)

    for text, (x, y) in labels:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Ramka na ekranie (żółta)
    cv2.rectangle(frame, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 255, 255), 2)

    # Zapisz klatkę
    output_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(output_path, frame)
    frame_count += 1

    # Wyświetl
    cv2.imshow("output", frame)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
