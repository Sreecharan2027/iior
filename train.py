import subprocess
import time
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Enable mixed precision if GPU supports it
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print("✅ Mixed precision enabled")
except:
    print("⚠️ Mixed precision not used")

# Clean up any previous session
clear_session()

# ---------------- TIMING START ---------------- #
start_time = time.time()

# ---------------- CALLBACKS ---------------- #
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# ---------------- MODEL ARCHITECTURE ---------------- #
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Unfreeze top 30 layers only (gradual fine-tuning)
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output_layer = Dense(30, activation="softmax")(x)  # 30 classes

model = Model(inputs=base_model.input, outputs=output_layer)
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# ---------------- DATASET ---------------- #
# Assuming you have numpy arrays or you can adapt this to use ImageDataGenerator

# If using raw images:
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
#                              rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
#                              horizontal_flip=True, zoom_range=0.2)

# train_generator = datagen.flow_from_directory('path/to/data', target_size=(299, 299),
#                                               batch_size=32, class_mode='sparse', subset='training')
# val_generator = datagen.flow_from_directory('path/to/data', target_size=(299, 299),
#                                             batch_size=32, class_mode='sparse', subset='validation')

# OR if using arrays directly (as in your code)
# Normalize X_train and X_val
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,  # Batch size increased from 4 to 32
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# ---------------- GPU STATS ---------------- #
def get_gpu_stats():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw', '--format=csv,noheader,nounits']
        ).decode('utf-8').strip()
        stats = output.split(',')
        return {
            'GPU Name': stats[0].strip(),
            'Memory Used (MB)': f"{stats[1].strip()} / {stats[2].strip()}",
            'GPU Load (%)': stats[3].strip(),
            'Memory Load (%)': stats[4].strip(),
            'Power Usage (W)': stats[5].strip()
        }
    except Exception as e:
        return {"GPU Stats": "Not available (CPU used)", "Error": str(e)}

gpu_stats = get_gpu_stats()

# ---------------- RESULTS ---------------- #
end_time = time.time()
elapsed_time = end_time - start_time

final_epoch = len(history.history['accuracy'])
final_acc = history.history['accuracy'][-1]
final_loss = history.history['loss'][-1]

print("\n📊 GPU Metrics:")
for k, v in gpu_stats.items():
    print(f"  {k}: {v}")

print("\n🏁 Training Summary:")
print(f"  Total Epochs: {final_epoch}")
print(f"  Final Loss: {final_loss:.4f}")
print(f"  Final Accuracy: {final_acc:.4f}")

print("\n⏱️ Timing:")
print(f"  Total Training Time: {elapsed_time:.2f} seconds")
