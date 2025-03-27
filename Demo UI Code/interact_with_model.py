from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Set up logging to see what’s happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthSeekerInference:
    def __init__(self, model_path):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        # Use CPU for simplicity, change to "cuda" if you’ve got a GPU
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def predict(self, text):
        """Make a prediction on the input text."""
        logger.info(f"Predicting: {text}")
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item() - 1  # Shift [0,1,2] to [-1,0,1]
                label_map = {-1: "False", 0: "Mixed", 1: "True"}
                return label_map[prediction]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"Error: {str(e)}"

def interact_with_model(model_path):
    """Interactive loop to test the model in the terminal."""
    inference = TruthSeekerInference(model_path)
    print("\nWelcome to TruthSeeker LLM Tester!")
    print("Type a statement to analyze its truthfulness. Enter 'quit' to exit.\n")
    
    while True:
        statement = input("Enter statement: ").strip()
        if statement.lower() == 'quit':
            print("Exiting...")
            break
        if not statement:
            print("Please enter a statement.")
            continue
        
        prediction = inference.predict(statement)
        print(f"Prediction: {prediction}\n")

if __name__ == "__main__":
    # Update this path if your model is somewhere else
    model_path = "/Users/mattlaing/Desktop/TruthSeeker/models/finetuned"  # Relative path from project root
    # Or use absolute path, e.g., "/Users/mattlaing/Desktop/TruthSeeker/models/finetuned"
    
    try:
        interact_with_model(model_path)
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        print(f"Error: Couldn’t start the tester. Check logs for details.")
