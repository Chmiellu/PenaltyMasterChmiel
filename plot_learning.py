import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Dodane do zmiany osi X

# Wczytaj dane z CSV
log_path = 'training_log.csv'
history = pd.read_csv(log_path)

# Wykres strat
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Całkowite liczby na osi X

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Całkowite liczby na osi X

plt.tight_layout()
plt.show()
