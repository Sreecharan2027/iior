import time
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# ---------------- CPU MODE ---------------- #
tf.config.set_visible_devices([], 'GPU')  # Ensure only CPU is used
print("✅ Using CPU for training")

# ---------------- TIMING START ---------------- #
start_time = time.time()

# ---------------- CALLBACKS ---------------- #
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_cpu.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# ---------------- MODEL ---------------- #
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Freeze most layers for CPU efficiency
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output_layer = Dense(30, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ---------------- DATA ---------------- #
# Assuming X_train, y_train, X_val, y_val are ready and normalized to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0

# ---------------- TRAIN ---------------- #
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,  # Reduced batch size for CPU
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    shuffle=True,
    verbose=1,
    workers=4,
    use_multiprocessing=True
)

# ---------------- FINAL REPORT ---------------- #
end_time = time.time()
elapsed_time = end_time - start_time

final_epoch = len(history.history['accuracy'])
final_acc = history.history['accuracy'][-1]
final_loss = history.history['loss'][-1]

print("\n🏁 Training Summary:")
print(f"  Total Epochs: {final_epoch}")
print(f"  Final Loss: {final_loss:.4f}")
print(f"  Final Accuracy: {final_acc:.4f}")
print(f"\n⏱️ Total Training Time: {elapsed_time / 60:.2f} minutes")



from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# normalize : 
from tensorflow.keras.applications.inception_v3 import preprocess_input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
#Rescaling from [0,255] to [-1, 1] (specific to InceptionV3)



#with fine tune
import time
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --------- CONFIG --------- #
num_classes = 50
input_shape = (224, 224, 3)

# Force CPU usage (optional)
tf.config.set_visible_devices([], 'GPU')
print("✅ Using CPU for training")

# --------- TIMING START --------- #
start_time = time.time()

# --------- CALLBACKS --------- #
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_mobilenet_finetuned.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)

# --------- LOAD BASE MODEL --------- #
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# --------- CUSTOM HEAD --------- #
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# --------- COMPILE & PRETRAIN --------- #
model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --------- DATA PREPROCESSING --------- #
X_train = tf.image.resize(X_train, (224, 224))
X_val = tf.image.resize(X_val, (224, 224))

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# --------- PRETRAIN --------- #
print("\n🔧 Phase 1: Training head only...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    shuffle=True,
    verbose=1,
    workers=4,
    use_multiprocessing=True
)

# --------- FINE-TUNING --------- #
print("\n🔁 Phase 2: Fine-tuning entire model...")

# Unfreeze all layers
for layer in base_model.layers:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Continue training (fine-tuning)
history_finetune = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    shuffle=True,
    verbose=1,
    workers=4,
    use_multiprocessing=True
)

# --------- RESULTS --------- #
end_time = time.time()
elapsed_time = end_time - start_time

final_epoch = len(history_finetune.history['accuracy'])
final_acc = history_finetune.history['accuracy'][-1]
final_loss = history_finetune.history['loss'][-1]

print("\n🏁 Fine-Tuning Summary:")
print(f"  Epochs Trained: {final_epoch}")
print(f"  Final Loss: {final_loss:.4f}")
print(f"  Final Accuracy: {final_acc:.4f}")
print(f"\n⏱️ Total Training Time: {elapsed_time / 60:.2f} minutes")

