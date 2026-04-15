"""
Flask Backend – CVM Skeletal Age Prediction
Imports pipeline directly from unknown_pred.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from unknown_pred import predict_age

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        result = predict_age(file.read())
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    from unknown_pred import DEVICE
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(debug=True, port=8000, use_reloader=False)