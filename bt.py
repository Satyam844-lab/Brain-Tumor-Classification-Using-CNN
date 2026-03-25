import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# dataset paths
train_dir = r"C:\Users\satya\Desktop\Training"
test_dir = r"C:\Users\satya\Desktop\Testing"

img_size = 128
batch_size = 32

# load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(

    train_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(

    test_dir,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

# class names
class_names = train_data.class_names

print("Classes:", class_names)


# normalize pixel values (0–255 → 0–1)
normalization_layer = layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))


# CNN model
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(4, activation='softmax')   # 4 output classes
])


# compile
model.compile(

    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# train
history = model.fit(

    train_data,
    epochs=15
)


# evaluate
loss, acc = model.evaluate(test_data)

print("Test Accuracy:", acc)

model.save("brain_tumor_4classes.keras")