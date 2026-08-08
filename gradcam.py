import tensorflow as tf
import numpy as np
import cv2
import os

# -------------------------------------------------
# Load model once
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "my_cnn_best.keras")

model = tf.keras.models.load_model(MODEL_PATH)

LAST_CONV_LAYER = "conv2d_1"


# -------------------------------------------------
# Generate Grad-CAM
# -------------------------------------------------

def generate_gradcam(image_path,
                     heatmap_save_path,
                     overlay_save_path):

    # ------------------------------
    # Load image
    # ------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224,224)
    )

    image = tf.keras.utils.img_to_array(image)

    image_batch = np.expand_dims(image, axis=0)

    # ------------------------------
    # Build Grad-CAM model
    # ------------------------------

    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(LAST_CONV_LAYER).output
    )

    classifier_input = tf.keras.Input(
        shape=feature_extractor.output.shape[1:]
    )

    x = classifier_input

    for layer_name in [
        "max_pooling2d_1",
        "flatten",
        "dense",
        "dropout",
        "dense_1"
    ]:
        x = model.get_layer(layer_name)(x)

    classifier = tf.keras.Model(classifier_input, x)

    # ------------------------------
    # Compute gradients
    # ------------------------------

    with tf.GradientTape() as tape:

        conv_features = feature_extractor(image_batch)

        tape.watch(conv_features)

        predictions = classifier(conv_features)

        class_index = tf.argmax(predictions[0])

        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_features)

    pooled_grads = tf.reduce_mean(
        gradients,
        axis=(0,1,2)
    )

    conv_features = conv_features[0].numpy()

    pooled_grads = pooled_grads.numpy()

    for i in range(conv_features.shape[-1]):
        conv_features[:,:,i] *= pooled_grads[i]

    heatmap = np.mean(conv_features, axis=-1)

    heatmap = np.maximum(heatmap,0)

    if heatmap.max()>0:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(
        heatmap,
        (224,224)
    )

    heatmap_uint8 = np.uint8(255*heatmap)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(
        heatmap_save_path,
        colored_heatmap
    )

    original = cv2.imread(image_path)

    original = cv2.resize(
        original,
        (224,224)
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    cv2.imwrite(
        overlay_save_path,
        overlay
    )

    return heatmap_save_path, overlay_save_path