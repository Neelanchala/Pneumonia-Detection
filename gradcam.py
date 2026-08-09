import tensorflow as tf
import numpy as np
import cv2
import os


# =========================================================
# Load Model
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "my_cnn_best.keras"
)

print("Loading model for Grad-CAM...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Grad-CAM model loaded.")


# =========================================================
# MobileNetV2 Layer
# =========================================================

MOBILENET_LAYER = "mobilenetv2_1.00_224"

LAST_CONV_LAYER = "block_16_project"

mobilenet = model.get_layer(MOBILENET_LAYER)

target_layer = mobilenet.get_layer(LAST_CONV_LAYER)


# =========================================================
# Grad-CAM
# =========================================================

def generate_gradcam(
    image_path,
    heatmap_save_path,
    overlay_save_path
):

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # Apply the same Rescaling used by the model
    # -----------------------------------------------------

    rescaled_image = model.layers[1](
        image_batch
    )


    # -----------------------------------------------------
    # Build feature model INSIDE MobileNetV2
    # -----------------------------------------------------

    feature_model = tf.keras.models.Model(
        inputs=mobilenet.input,
        outputs=[
            target_layer.output,
            mobilenet.output
        ]
    )


    # -----------------------------------------------------
    # Calculate gradients
    # -----------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, mobilenet_output = feature_model(
            rescaled_image,
            training=False
        )

        # GlobalAveragePooling
        x = model.layers[3](
            mobilenet_output
        )

        # Dropout
        x = model.layers[4](
            x,
            training=False
        )

        # Final Dense layer
        predictions = model.layers[5](
            x
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    # -----------------------------------------------------
    # Gradients
    # -----------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )


    # -----------------------------------------------------
    # Global average pooling of gradients
    # -----------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    # -----------------------------------------------------
    # Get feature map
    # -----------------------------------------------------

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients.numpy()

    conv_outputs = conv_outputs.numpy()


    # -----------------------------------------------------
    # Weight feature maps
    # -----------------------------------------------------

    for i in range(
        conv_outputs.shape[-1]
    ):

        conv_outputs[:, :, i] *= (
            pooled_gradients[i]
        )


    # -----------------------------------------------------
    # Generate heatmap
    # -----------------------------------------------------

    heatmap = np.mean(
        conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )


    if np.max(heatmap) > 0:

        heatmap /= np.max(
            heatmap
        )


    # -----------------------------------------------------
    # Resize heatmap
    # -----------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )


    # -----------------------------------------------------
    # Convert to color heatmap
    # -----------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )


    # -----------------------------------------------------
    # Save heatmap
    # -----------------------------------------------------

    cv2.imwrite(
        heatmap_save_path,
        colored_heatmap
    )


    # -----------------------------------------------------
    # Load original image
    # -----------------------------------------------------

    original = cv2.imread(
        image_path
    )

    original = cv2.resize(
        original,
        (224, 224)
    )


    # -----------------------------------------------------
    # Create overlay
    # -----------------------------------------------------

    overlay = cv2.addWeighted(
        original,
        0.6,
        colored_heatmap,
        0.4,
        0
    )


    # -----------------------------------------------------
    # Save overlay
    # -----------------------------------------------------

    cv2.imwrite(
        overlay_save_path,
        overlay
    )


    return (
        heatmap_save_path,
        overlay_save_path
    )