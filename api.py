from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import TruthSeekerInference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "http://localhost:8080"}})

model_path = "/Users/mattlaing/Desktop/TruthSeeker/models/finetuned"  # Update this path as needed
inference = TruthSeekerInference(model_path)

@app.route('/analyze', methods=['POST'])
def analyze_statement():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        prediction = inference.predict(text)
        return jsonify({"statement": text, "prediction": prediction}), 200
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)