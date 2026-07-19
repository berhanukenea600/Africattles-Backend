from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)
from flask_cors import CORS

CORS(app, resources={r"/*": {"origins": "*"}})

# Load the AI model once when the server starts
model = tf.keras.models.load_model("africattles_weight_model.keras")

IMG_SIZE = (224, 224)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "app": "AfriCattles Backend",
        "version": "1.0"
    })

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({
        "predicted_weight": 180,
        "unit": "kg"
    })
@app.route("/test")
def test():
    return jsonify({"message": "Backend works!"})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
