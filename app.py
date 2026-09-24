from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Limit TensorFlow CPU threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
CORS(app)

# Load AI model once
model = tf.keras.models.load_model("africattles_weight_model.keras")

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("MODEL SUMMARY:", flush=True)
model.summary()

IMG_SIZE = (224, 224)


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "app": "AfriCattles Backend",
        "version": "1.0"
    })


@app.route("/test")
def test():
    return jsonify({
        "message": "Backend works!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("STEP 1", flush=True)

        if "image" not in request.files:
            print("NO IMAGE", flush=True)
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        print("STEP 2 - image received", flush=True)

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        print("STEP 3 - image resized", flush=True)

        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        print("STEP 4 - preprocessing finished", flush=True)

        print("STARTING MODEL INFERENCE", flush=True)

        prediction = model(img, training=False)

        print("MODEL INFERENCE FINISHED", flush=True)

        weight = float(prediction.numpy()[0][0])

        # Meat-based reference valuation
        meat_yield_low = 0.40
        meat_yield_high = 0.45
        meat_price_per_kg = 1700

        meat_value_low = (
            weight * meat_yield_low * meat_price_per_kg
        )

        meat_value_high = (
            weight * meat_yield_high * meat_price_per_kg
        )

        print("PREDICTED WEIGHT:", weight, flush=True)
        print("ESTIMATED MEAT VALUE LOW:", meat_value_low, flush=True)
        print("ESTIMATED MEAT VALUE HIGH:", meat_value_high, flush=True)

        return jsonify({
            "predicted_weight": weight,
            "estimated_meat_value_low": meat_value_low,
            "estimated_meat_value_high": meat_value_high
        })

    except Exception as e:
        import traceback

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
        )
