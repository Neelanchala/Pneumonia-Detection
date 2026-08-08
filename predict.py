import tensorflow as tf
import numpy as np
import os

print("Loading model...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "my_cnn_best.keras")

print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

CLASS_NAMES = ["Normal", "Pneumonia"]


def predict_image(image_path):

    print("Loading image...")
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))

    print("Converting to array...")
    image = tf.keras.utils.img_to_array(image)

    image = np.expand_dims(image, axis=0)

    print("Running prediction...")
    prediction = model.predict(image, verbose=0)

    print("Prediction complete.")

    class_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    predicted_class = CLASS_NAMES[class_index]

    return predicted_class, confidence  