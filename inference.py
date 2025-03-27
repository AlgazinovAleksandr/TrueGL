from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthSeekerInference:
    def __init__(self, model_path):
        logger.info(f"Loading model from {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def predict(self, text):
        logger.info(f"Predicting: {text}")
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item() - 1  # Shift [0,1,2] to [-1,0,1]
                label_map = {-1: "False", 0: "Mixed", 1: "True"}
                return label_map[prediction]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

# Add to bottom of src/inference.py if not there
if __name__ == "__main__":
    model_path = "models/finetuned"  # Relative path for portability
    inference = TruthSeekerInference(model_path)
    print(inference.predict("The sky is blue."))