from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load AI model once
model = tf.keras.models.load_model("africattles_weight_model.keras")

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
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Read image
        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)

        # Preprocess
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # AI prediction
        prediction = model.predict(img, verbose=0)

        weight = round(float(prediction[0][0]), 1)

        return jsonify({
            "predicted_weight": weight,
            "unit": "kg"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
