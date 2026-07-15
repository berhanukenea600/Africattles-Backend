from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)

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

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    weight = round(float(prediction[0][0]), 1)

    return jsonify({
        "predicted_weight": weight,
        "unit": "kg"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
