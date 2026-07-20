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
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
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

    print("PREDICT ENDPOINT HIT")

    return jsonify({
        "predicted_weight": 999
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
