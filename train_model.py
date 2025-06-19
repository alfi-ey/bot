import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("dataagr_clean.csv")
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 4. Build model
model = Sequential([
    Dense(128, activation='relu', input_dim=X.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,          # berhenti setelah 10 epoch tanpa peningkatan
    min_delta=0.0005,     # toleransi perbedaan minimal
    restore_best_weights=True
)

# 6. Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model selesai dilatih hingga epoch ke-{len(history.history['loss'])}!")
print(f"🔍 Akurasi Validasi Akhir: {acc:.4f}")
print(f"📉 Final Loss: {loss:.4f}")

# 8. Report
y_pred = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

# 9. Save model & preprocessing tools
model.save("trained_model.keras")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n📁 Model dan alat praproses berhasil disimpan:")
print("- trained_model.keras")
print("- scaler.pkl")
print("- label_encoder.pkl")

# 10. Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_performance.png")
plt.show()

# 11. Predict manual
def predict_crop_backend(N, P, K, temperature, humidity, ph, rainfall):
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    print("\n🌱 HASIL PREDIKSI:")
    print(f"Tanaman yang Direkomendasikan: {predicted_label.upper()}")
    print(f"Tingkat Keyakinan: {confidence:.2%}")

# 12. Manual test
if __name__ == "__main__":
    print("\n🧪 Tes Prediksi Manual")
    try:
        N = float(input("Input Nitrogen (N): "))
        P = float(input("Input Phosphorus (P): "))
        K = float(input("Input Kalium (K): "))
        temperature = float(input("Input Temperature (°C): "))
        humidity = float(input("Input Humidity (%): "))
        ph = float(input("Input pH tanah: "))
        rainfall = float(input("Input Rainfall (mm): "))

        predict_crop_backend(N, P, K, temperature, humidity, ph, rainfall)

    except ValueError:
        print("⚠️ Input salah! Harus berupa angka.")
