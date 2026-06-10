import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = r"D:\FINAL PRJ\dataset_split_balanced"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

MODEL_PATH = r"D:\FINAL PRJ\drowsiness_mobilenetv2.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
)

callbacks = [checkpoint, early_stopping, reduce_lr]


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Train classes:", train_gen.class_indices)
print("Val classes:", val_gen.class_indices)
print("Test classes:", test_gen.class_indices)

print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)
print("Test samples:", test_gen.samples)


if os.path.exists(MODEL_PATH):
    best_model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded existing model from:", MODEL_PATH)
else:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(MODEL_PATH)
    print("Trained and loaded best model from:", MODEL_PATH)


test_loss, test_acc = best_model.evaluate(test_gen)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

y_prob = best_model.predict(test_gen)
y_pred = (y_prob > 0.5).astype(int).reshape(-1)
y_true = test_gen.classes

print("Class mapping:", test_gen.class_indices)
print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(test_gen.class_indices.keys()),
    )
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(test_gen.class_indices.keys()),
    yticklabels=list(test_gen.class_indices.keys()),
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Eye State Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(r"D:\FINAL PRJ\eye_model_confusion_matrix.png")
plt.show()


if "history" in globals():
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Phase 1 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Phase 1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(r"D:\FINAL PRJ\eye_model_training_curve.png")
    plt.show()
else:
    print("No training history found. Skip training curve.")
