from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "VetDerma AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    animal = request.form.get("animal")

    if animal == "cow":
        result = {
            "disease": "Lumpy Skin Disease (LSD)",
            "confidence": 95,
            "severity": "Severe"
        }
    else:
        result = {
            "disease": "Pyoderma",
            "confidence": 92,
            "severity": "Moderate"
        }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)