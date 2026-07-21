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
    try:
        print("STEP 1")

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        print("STEP 2")

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        print("STEP 3")

        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        print("STEP 4")

        prediction = model.predict(img, verbose=0)

        print("STEP 5")
        print(prediction)

        weight = float(prediction[0][0])

        return jsonify({
            "predicted_weight": weight
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
