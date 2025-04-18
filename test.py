import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to hide INFO messages as well

# Optionally, set TensorFlow's own logger level to ERROR
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# For Transformers, reduce its verbosity:
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Optionally, ignore Python warnings
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from transformers import AutoTokenizer
import argparse


model_loc = "saved_model/xlm-r_88acc_best_model_v2"
model_name = "xlm-roberta-base"

def inference(input_text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the input text
    encoded_inputs = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True)
    
    # dummy token_type_ids for tokenizing
    dummy_token_type_ids = tf.zeros_like(encoded_inputs["input_ids"])
    
    # Prepare the input dictionary
    inputs = {
        "input_ids": encoded_inputs["input_ids"],
        "attention_mask": encoded_inputs["attention_mask"],
        "token_type_ids": dummy_token_type_ids,
    }
    
    # Load the SavedModel
    loaded_model = tf.saved_model.load(model_loc)
    infer = loaded_model.signatures["serving_default"]
    
    # Run inference
    outputs = infer(**inputs)
    
    # Extract logits
    logits = outputs["logits"]
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(logits)
    
    # Get the predicted class index
    predicted_class_index = tf.argmax(probabilities, axis=1).numpy()[0]
    
    # Map class index to label
    class_labels = {0: "HAM", 1: "SCAM"}
    predicted_label = class_labels[predicted_class_index]
    
    # Output the result
    print("Message: " + input_text)
    print(f"The message is classified as: {predicted_label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    print(f"Text Message: {args.text}")
    
    inference(args.text)