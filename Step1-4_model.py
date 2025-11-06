# =============================================================
# AER 850 Project 2 — TWO-MODEL CNN TRAINING + EVALUATION
# WITH FORMATTED MODEL SUMMARY TABLE
# =============================================================

import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

from tabulate import tabulate   # NEW

# =============================================================
# CONFIG
# =============================================================
IMG_SHAPE = (500, 500, 3)
EPOCHS = 30
DATA_DIR = "Data"

TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR   = f"{DATA_DIR}/valid"
TEST_DIR  = f"{DATA_DIR}/test"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =============================================================
# MODEL SUMMARY TABLE   
# =============================================================
def model_summary_table(model, model_name):
    layers = []
    for layer in model.layers:
        name = layer.name
        layer_type = layer.__class__.__name__
        output_shape = layer.output_shape if hasattr(layer, "output_shape") else "---"
        params = layer.count_params()
        layers.append([name, layer_type, str(output_shape), params])

    print(f"\nMODEL SUMMARY TABLE — {model_name}\n")
    print(tabulate(
        layers,
        headers=["Layer Name", "Layer Type", "Output Shape", "Param #"],
        tablefmt="fancy_grid"
    ))

# =============================================================
# DATA LOADER
# =============================================================
def load_data(batch_size):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.25, zoom_range=0.25,
        horizontal_flip=True, vertical_flip=True,
        rotation_range=30,
        width_shift_range=0.25, height_shift_range=0.25,
        brightness_range=[0.85,1.15]
    ).flow_from_directory(
        TRAIN_DIR, target_size=IMG_SHAPE[:2], batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR, target_size=IMG_SHAPE[:2], batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    return train_gen, val_gen, train_gen.class_indices

# =============================================================
# MODEL BUILDER
# =============================================================
def create_model(dropout_rate):
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=IMG_SHAPE), MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'), MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation='relu'), MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation='relu'), MaxPooling2D(2,2),
        Flatten(),
        Dense(256,activation='relu'), Dropout(dropout_rate),
        Dense(128,activation='relu'), Dropout(dropout_rate),
        Dense(3,activation='softmax')
    ])
    return model

# =============================================================
# TRAINING FUNCTION
# =============================================================
def train_model(model, train_gen, val_gen, lr, label):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS, callbacks=callbacks, verbose=1
    )

    model.save(f"models/{label}.keras")

    pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
    plt.title(f"Accuracy — {label}"); plt.grid()
    plt.savefig(f"outputs/{label}_accuracy.png"); plt.close()

    pd.DataFrame(history.history)[['loss','val_loss']].plot()
    plt.title(f"Loss — {label}"); plt.grid()
    plt.savefig(f"outputs/{label}_loss.png"); plt.close()

    return history.history

# =============================================================
# EVALUATION FUNCTION
# =============================================================
def evaluate_model(model, class_dict, label):
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        TEST_DIR, target_size=IMG_SHAPE[:2], batch_size=1,
        class_mode='categorical', shuffle=False
    )

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=list(class_dict.keys()),
                yticklabels=list(class_dict.keys()))
    plt.title(f"Confusion Matrix — {label}")
    plt.savefig(f"outputs/{label}_confusion_matrix.png")
    plt.close()

    print(f"\nClassification Report — {label}:\n")
    print(classification_report(y_true, y_pred, target_names=list(class_dict.keys())))

    encoder = OneHotEncoder(sparse_output=False).fit(y_true.reshape(-1,1))
    y_true_ohe = encoder.transform(y_true.reshape(-1,1))
    roc = roc_auc_score(y_true_ohe, preds, multi_class="ovr")
    print(f"ROC-AUC Score ({label}): {roc:.4f}\n")

# =============================================================
# MAIN — TRAIN TWO MODELS
# =============================================================
if __name__ == "__main__":

    hp_A = {"lr":0.0005, "batch":32, "drop":0.5, "label":"Model_A_5e4_drop50"}
    hp_B = {"lr":0.0001, "batch":32, "drop":0.3, "label":"Model_B_1e4_drop30"}

    train_A, val_A, class_dict = load_data(hp_A["batch"])
    model_A = create_model(hp_A["drop"])
    model_summary_table(model_A, hp_A["label"])   
    train_model(model_A, train_A, val_A, hp_A["lr"], hp_A["label"])
    evaluate_model(model_A, class_dict, hp_A["label"])

    train_B, val_B, _ = load_data(hp_B["batch"])
    model_B = create_model(hp_B["drop"])
    model_summary_table(model_B, hp_B["label"])   
    train_model(model_B, train_B, val_B, hp_B["lr"], hp_B["label"])
    evaluate_model(model_B, class_dict, hp_B["label"])

    print("\nTraining & Evaluation Complete\n")
