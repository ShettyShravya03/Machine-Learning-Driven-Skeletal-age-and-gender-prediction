"""
app.py  –  Flask Backend for CVM Skeletal Age & Gender Prediction

Thin HTTP layer — all the actual ML logic lives in unknown_pred.py.
This file just handles routing, error formatting, and CORS.

Keeping it thin was a deliberate choice — early version had
preprocessing and model loading here too, which made testing
a nightmare. Moved everything to unknown_pred.py so the
prediction pipeline can be tested independently of Flask.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from unknown_pred import predict_age, DEVICE

app = Flask(__name__)
CORS(app)
# CORS needed because React frontend runs on port 3000
# and Flask runs on 8000 — different origins = blocked by browser
# without this header


# =====================================================================
#  PREDICTION ENDPOINT
# =====================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data POST with an X-ray image file.
    Returns skeletal age, gender, confidence, and top SHAP factors.

    Errors are separated by type:
      400 = client sent bad request (no file, empty filename)
      422 = file was received but prediction failed (bad image format,
            fewer than 3 vertebrae detected, etc.)
      500 = unexpected server error — logged with full traceback
    """
    # validate file presence before doing any work
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        # browser sent the field but didn't actually attach a file
        return jsonify({"error": "Empty filename"}), 400

    try:
        # file.read() passes raw bytes — unknown_pred handles
        # decoding, preprocessing, segmentation, and prediction
        result = predict_age(file.read())
        return jsonify(result)

    except ValueError as e:
        # ValueError = known failure from our pipeline
        # e.g. "fewer than 3 vertebrae detected"
        # 422 not 500 — server worked fine, input was the problem
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        # unexpected error — log full traceback server-side
        # only return the message to client, not the stack trace
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =====================================================================
#  HEALTH CHECK ENDPOINT
# =====================================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Quick liveness check — React frontend pings this on load
    to confirm the backend is up before enabling the upload button.
    Also returns device so we know if GPU is available in deployment.
    """
    return jsonify({
        "status": "ok",
        "device": str(DEVICE)
        # "cuda" = GPU inference, "cpu" = slower but works
        # useful to know during demo if predictions feel slow
    })


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=8000,
        use_reloader=False
        # use_reloader=False because reloader loads the model twice
        # — once in the parent process, once in the child.
        # With a 200MB PyTorch model that adds 10 seconds to startup
        # and occasionally caused CUDA memory errors during dev.
    )