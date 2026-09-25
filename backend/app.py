from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
from PIL import Image
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

import disease_info
from disease_info import DOG_CLASSES, COW_CLASSES, DISEASE_INFO


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("\n========== DEBUG ==========")
print("Disease file:", disease_info.__file__)
print("DOG_CLASSES:", DOG_CLASSES)
print("COW_CLASSES:", COW_CLASSES)
print("DISEASE_INFO KEYS:", DISEASE_INFO.keys())
print("===========================\n")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("ERROR: MONGO_URI not found in .env file")
else:
    print("MongoDB URI loaded successfully")


# ============================================================
# MONGODB CONNECTION
# ============================================================

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    # Test connection
    client.admin.command("ping")

    print("MongoDB connected successfully!")

    # Database
    db = client["vetderma"]

    # Collections
    users_collection = db["users"]
    predictions_collection = db["predictions"]

except Exception as e:
    print("MongoDB connection failed:", e)
    client = None
    db = None
    predictions_collection = None


# ============================================================
# LOAD AI MODELS
# ============================================================

print("\nLoading AI models...")

cow_model = tf.keras.models.load_model("models/cow_model.keras")
print("Cow model loaded successfully!")

dog_model = tf.keras.models.load_model("models/dog_model.keras")
print("Dog model loaded successfully!")

print("All AI models loaded successfully!\n")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return "VetDerma AI Backend Running Successfully"


# ============================================================
# MONGODB TEST ROUTE
# ============================================================

@app.route("/db-test", methods=["GET"])
def db_test():

    if predictions_collection is None:
        return jsonify({
            "status": "error",
            "message": "MongoDB is not connected"
        }), 500

    try:

        client.admin.command("ping")

        return jsonify({
            "status": "success",
            "message": "MongoDB connection is working",
            "database": "vetderma",
            "collection": "predictions"
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
# ============================================================
# USER REGISTER API
# ============================================================

@app.route("/register", methods=["POST"])
def register():

    try:
        data = request.get_json()

        name = data.get("name", "").strip()
        role = data.get("role", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not name or not email or not password:
            return jsonify({
                "success": False,
                "message": "Please fill all required fields"
            }), 400

        if len(password) < 6:
            return jsonify({
                "success": False,
                "message": "Password must be at least 6 characters"
            }), 400

        existing_user = users_collection.find_one({
            "email": email
        })

        if existing_user:
            return jsonify({
                "success": False,
                "message": "Email already registered"
            }), 409

        hashed_password = generate_password_hash(password)

        user = {
            "name": name,
            "role": role,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.utcnow()
        }

        users_collection.insert_one(user)

        print("New user registered:", email)

        return jsonify({
            "success": True,
            "message": "Registration successful"
        }), 201

    except Exception as e:

        print("Registration error:", e)

        return jsonify({
            "success": False,
            "message": "Registration failed"
        }), 500


# ============================================================
# USER LOGIN API
# ============================================================

@app.route("/login", methods=["POST"])
def login():

    try:
        data = request.get_json()

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Please enter email and password"
            }), 400

        user = users_collection.find_one({
            "email": email
        })

        if not user:
            return jsonify({
                "success": False,
                "message": "Invalid email or password"
            }), 401

        if not check_password_hash(user["password"], password):
            return jsonify({
                "success": False,
                "message": "Invalid email or password"
            }), 401

        print("User logged in:", email)

        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": str(user["_id"]),
                "name": user["name"],
                "role": user.get("role", ""),
                "email": user["email"]
            }
        }), 200

    except Exception as e:

        print("Login error:", e)

        return jsonify({
            "success": False,
            "message": "Login failed"
        }), 500




# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    # --------------------------------------------------------
    # Get animal type
    # --------------------------------------------------------

    animal = request.form.get("animal")

    if animal not in ["cow", "dog"]:

        return jsonify({
            "error": "Invalid animal. Please select cow or dog."
        }), 400


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image = request.files["image"]

        image = Image.open(image).convert("RGB")

        image = preprocess_image(image)

    except Exception as e:

        return jsonify({
            "error": "Invalid image",
            "details": str(e)
        }), 400


    # ========================================================
    # COW PREDICTION
    # ========================================================

    if animal == "cow":

        prediction = cow_model.predict(image, verbose=0)

        print("\n========== COW MODEL OUTPUT ==========")

        for i, prob in enumerate(prediction[0]):

            print(
                f"{COW_CLASSES[i]} : {prob * 100:.2f}%"
            )

        print("======================================")


        index = np.argmax(prediction[0])

        confidence = float(
            np.max(prediction[0]) * 100
        )

        disease = COW_CLASSES[index]


    # ========================================================
    # DOG PREDICTION
    # ========================================================

    else:

        prediction = dog_model.predict(image, verbose=0)

        print("\n========== DOG MODEL OUTPUT ==========")

        for i, prob in enumerate(prediction[0]):

            print(
                f"{DOG_CLASSES[i]} : {prob * 100:.2f}%"
            )

        print("======================================")


        index = np.argmax(prediction[0])

        confidence = float(
            np.max(prediction[0]) * 100
        )

        disease = DOG_CLASSES[index]


    # ========================================================
    # UNKNOWN / LOW CONFIDENCE
    # ========================================================

    if confidence < 90:

        result = {

            "animal": animal,

            "disease": "Disease Not Found",

            "confidence": round(confidence, 2),

            "severity": "Unknown",

            "symptoms": [
                "Disease not available in trained dataset."
            ],

            "causes": [
                "Unknown skin condition."
            ],

            "treatment": [
                "Consult a veterinarian."
            ],

            "recommendation": [
                "Please upload a clearer image or consult a vet."
            ]
        }

        # ----------------------------------------------------
        # Save low-confidence prediction too
        # ----------------------------------------------------

        if predictions_collection is not None:

            try:

                prediction_record = {

                    "animal": animal,

                    "disease": "Disease Not Found",

                    "confidence": round(confidence, 2),

                    "severity": "Unknown",

                    "symptoms": result["symptoms"],

                    "causes": result["causes"],

                    "treatment": result["treatment"],

                    "recommendation": result["recommendation"],

                    "created_at": datetime.now(timezone.utc)
                }

                predictions_collection.insert_one(
                    prediction_record
                )

                print("Low-confidence prediction saved to MongoDB!")

            except Exception as e:

                print(
                    "MongoDB save failed:",
                    e
                )


        return jsonify(result)


    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    if disease not in DISEASE_INFO:

        return jsonify({

            "error": "Disease information not found",

            "disease": disease

        }), 500


    info = DISEASE_INFO[disease]


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "animal": animal,

        "disease": disease,

        "confidence": round(confidence, 2),

        "severity": info["severity"],

        "symptoms": info["symptoms"],

        "causes": info["causes"],

        "treatment": info["treatment"],

        "recommendation": info["recommendation"]
    }


    # ========================================================
    # SAVE PREDICTION TO MONGODB
    # ========================================================

    if predictions_collection is not None:

        try:

            prediction_record = {

                "animal": animal,

                "disease": disease,

                "confidence": round(confidence, 2),

                "severity": info["severity"],

                "symptoms": info["symptoms"],

                "causes": info["causes"],

                "treatment": info["treatment"],

                "recommendation": info["recommendation"],

                "created_at": datetime.now(timezone.utc)
            }


            predictions_collection.insert_one(
                prediction_record
            )

            print(
                "Prediction saved to MongoDB successfully!"
            )

        except Exception as e:

            print(
                "MongoDB save failed:",
                e
            )


    # ========================================================
    # DISPLAY RESULT IN TERMINAL
    # ========================================================

    print("\n========== AI PREDICTION ==========")

    print(result)

    print("===================================\n")


    # ========================================================
    # RETURN RESULT TO FRONTEND
    # ========================================================

    return jsonify(result)


# ============================================================
# RUN FLASK
# ============================================================

    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)   
 )