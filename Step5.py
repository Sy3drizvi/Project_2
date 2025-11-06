import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==============================================
# STEP 5: Real Image Visual Testing 
# ==============================================

# Load Models
model_A = keras.models.load_model("models/Model_A_5e4_drop50.keras")
model_B = keras.models.load_model("models/Model_B_1e4_drop30.keras")

# Class labels
class_names = ["crack", "missing-head", "paint-off"]

# Test image paths
img_paths = {
    "paint-off":   "Data/test/paint-off/test_paintoff.jpg",
    "missing-head":"Data/test/missing-head/test_missinghead.jpg",
    "crack":       "Data/test/crack/test_crack.jpg",
}

IMG_SHAPE = (500, 500)

# Output folder
save_dir = "outputs/step5_predictions"
os.makedirs(save_dir, exist_ok=True)

# Preprocess image
def load_and_prep(path):
    img = load_img(path, target_size=IMG_SHAPE)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Predict function
def predict(model, img):
    pred = model.predict(load_and_prep(img), verbose=0)[0]
    return pred, np.argmax(pred)

# Plot + Save function
def evaluate_and_save(model, model_name):
    for true_class, img_path in img_paths.items():
        img = load_img(img_path)
        probs, pred_idx = predict(model, img_path)
        pred_class = class_names[pred_idx]

        # green = correct prediction, red = wrong prediction
        is_correct = (pred_class == true_class)
        box_color = (0, 1, 0, 0.35) if is_correct else (1, 0, 0, 0.35)

        text = "\n".join([f"{c}: {p*100:.2f}%" for c, p in zip(class_names, probs)]) + \
               f"\nPredicted: {pred_class}\nTrue: {true_class}"

        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{model_name} — {true_class}")

        plt.text(
            0.02, 0.02, text, transform=plt.gca().transAxes,
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.4", fc=box_color, ec="none")
        )

        # Save image
        filename = f"{model_name}_{true_class}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")

# Run for both models
evaluate_and_save(model_A, "Model_A")
evaluate_and_save(model_B, "Model_B")

print(f"\nStep 5 completed. Images saved in '{save_dir}'")
