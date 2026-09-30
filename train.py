import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# STEP 1 — SETTINGS
# ─────────────────────────────────────────

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 30
SEED       = 42

TRAIN_DIR  = "/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri/Training"
TEST_DIR   = "/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri/Testing"

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────
# STEP 2 — LOAD DATA
# ─────────────────────────────────────────

# Training + Validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels        = "inferred",
    label_mode    = "categorical",   # one-hot for 4 classes
    image_size    = IMG_SIZE,
    batch_size    = BATCH_SIZE,
    validation_split = 0.20,
    subset        = "training",
    seed          = SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels        = "inferred",
    label_mode    = "categorical",
    image_size    = IMG_SIZE,
    batch_size    = BATCH_SIZE,
    validation_split = 0.20,
    subset        = "validation",
    seed          = SEED,
)

# Test set — separate folder, no split needed
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels        = "inferred",
    label_mode    = "categorical",
    image_size    = IMG_SIZE,
    batch_size    = BATCH_SIZE,
    shuffle       = False,           # keep order for evaluation
    seed          = SEED,
)

# Class names
class_names = train_ds.class_names
print(f"Classes: {class_names}")

# ─────────────────────────────────────────
# STEP 3 — NORMALIZE + PREFETCH
# ─────────────────────────────────────────

def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(normalize).cache().shuffle(500, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.map(normalize).cache().prefetch(AUTOTUNE)
test_ds  = test_ds.map(normalize).cache().prefetch(AUTOTUNE)

# ─────────────────────────────────────────
# STEP 4 — MODEL
# ─────────────────────────────────────────

model = models.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', 
                  padding='same', input_shape=(128,128,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    # Block 4 — extra depth for 4-class problem
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(4, activation='softmax')
])



# ─────────────────────────────────────────
# STEP 5 — COMPILE
# ─────────────────────────────────────────

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-4),
    loss      = "categorical_crossentropy",
    # multi-class loss
    metrics   = ["accuracy"]
)

# ─────────────────────────────────────────
# STEP 6 — CALLBACKS
# ─────────────────────────────────────────

cb_list = [
    callbacks.EarlyStopping(
        monitor             = "val_loss",
        patience            = 5,
        restore_best_weights= True,
        verbose             = 1
    ),
    callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-7,
        verbose  = 1
    ),
    callbacks.ModelCheckpoint(
        filepath      = "best_model.keras",
        monitor       = "val_accuracy",
        save_best_only= True,
        verbose       = 1
    ),
]

# ─────────────────────────────────────────
# STEP 7 — TRAIN
# ─────────────────────────────────────────

history = model.fit(
    train_ds,
    epochs          = EPOCHS,
    validation_data = val_ds,
    callbacks       = cb_list,
    verbose         = 1
)

# ─────────────────────────────────────────
# STEP 8 — EVALUATE
# ─────────────────────────────────────────

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Collect predictions and true labels
y_true, y_pred = [], []

for images, labels in test_ds:
    preds  = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds,   axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ─────────────────────────────────────────
# STEP 9 — PLOTS
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].plot(history.history["accuracy"],     label="Train", color="#378ADD")
axes[0].plot(history.history["val_accuracy"], label="Val",   color="#1D9E75", linestyle="--")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history.history["loss"],     label="Train", color="#D85A30")
axes[1].plot(history.history["val_loss"], label="Val",   color="#993C1D", linestyle="--")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
    xticklabels=class_names, yticklabels=class_names
)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
