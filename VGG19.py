import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models, optimizers

# -------------------------------------------
# 1. Download Dataset
# -------------------------------------------
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Dataset path:", path)

dataset_dir = path

# -------------------------------------------
# 2. Preprocess + Split (80/20)
# -------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 16   # smaller batch = faster on CPU

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=42
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=42
)

num_classes = train_gen.num_classes

# -------------------------------------------
# 3. Build VGG19
# -------------------------------------------
base_model = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.20),
    layers.Dense(num_classes, activation='softmax')
])

# -------------------------------------------
# 4. Compile
# -------------------------------------------
optimizer = optimizers.Adam(learning_rate=0.01)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------
# 5. Train
# -------------------------------------------
history = model.fit(
    train_gen,
    epochs=10,   # Good for CPU training
    validation_data=val_gen
)

print("Training finished successfully!")
