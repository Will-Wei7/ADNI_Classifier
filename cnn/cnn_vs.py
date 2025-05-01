import os 
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras 
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
folder_path = "F:/桌面/ECE 545/545 Project/AugmentedAlzheimerDataset"
filenames = os.listdir(folder_path)
images = []

for filename in filenames:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)
        images.append(img)       
folder_path = r"F:/桌面/ECE 545/545 Project/AugmentedAlzheimerDataset"

image_paths = []
labels = []


for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    label_name = subfolder

    for filename in os.listdir(subfolder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(subfolder_path, filename)
            image_paths.append(file_path)
            labels.append(label_name)
df = pd.DataFrame({'image': image_paths, 'label': labels})

print(df.head())
print("Total images:", len(df))
train_df, temp_df = train_test_split(
    df, 
    test_size=0.20,
    random_state=42,
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['label']
)

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
print("Test samples:", len(test_df))
train_datagen = ImageDataGenerator(
    rescale=1./255,
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
Size = (224, 224)
batch_size = 32

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image',
    y_col='label',
    target_size=Size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image',
    y_col='label',
    target_size=Size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image',
    y_col='label',
    target_size=Size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
from sklearn.utils import class_weight

classes = np.unique(train_df['label'])
cw = class_weight.compute_class_weight('balanced', classes=classes, y=train_df['label'])
class_weights = {train_gen.class_indices[c]: weight for c, weight in zip(classes, cw)}
print("Class weights:", class_weights)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
num_classes = len(train_gen.class_indices)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop]
)

for layer in base_model.layers[-1:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop]
)

for layer in base_model.layers[-2:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

training_history=model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop]
)



results = pd.DataFrame({
    "Training Accuracy": training_history.history['accuracy'],
    "Validation Accuracy": training_history.history['val_accuracy'],
    "Training Loss": training_history.history['loss'],
    "Validation Loss": training_history.history['val_loss']
})
test_loss, test_acc = model.evaluate(test_gen)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes 
class_labels = list(test_gen.class_indices.keys()) 
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=class_labels))

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
