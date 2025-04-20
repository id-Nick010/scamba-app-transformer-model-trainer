import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import os
import tensorflow as tf
import itertools
import gc
import mlflow

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random

import argparse

MAX_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 100

randnum = 10#42

mlflow.sklearn.autolog()

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    return df

# Tokenization function
def tokenize_data(texts, tokenizer):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Prepare datasets
def prepare_datasets(train_df, val_df, test_df, tokenizer):
    train_encodings = tokenize_data(train_df['text'].tolist(), tokenizer)
    val_encodings = tokenize_data(val_df['text'].tolist(), tokenizer)
    test_encodings = tokenize_data(test_df['text'].tolist(), tokenizer)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        train_df['label'].values
    )).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask']
        },
        val_df['label'].values
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        test_df['label'].values
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def model_abbre(model_name):
    cases = {
        "bert-base-uncased": "BertBase",
        "bert-base-multilingual-cased": "MBERT",
        'xlm-roberta-base': 'XLM-RoBERTa' 
        #'google-bert/bert-base-cased': 'mobileBert'
    }
    return cases.get(model_name, "Model Unavailable")
    
def run_training(hp, model_name):
    mlflow.set_experiment("Third Final Evaluation")
    run_name = f"{hp['exp_desc']}_{model_abbre(model_name)}__lr{hp['learning_rate']}_ep{hp['epochs']}_bs{hp['batch_size']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(hp)
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("model", model_name)
        print("||--------------------------------------||")        
        print(f"||===>> Starting run: {run_name} with hyperparameters: {hp}")
        print("||--------------------------------------||")        

        #Red Info Logs Killer
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        
        # Print the TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        # List available GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("GPUs detected:")
            for gpu in gpus:
                print(gpu)
        else:
            print("No GPUs detected.")
            
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        
            # Configuration
        MODEL_NAME = model_name # also for tokenizer
                                 # 'bert-base-uncased' (bert)
                                 # 'bert-base-multilingual-cased' (mBERT)
                                 # 'xlm-roberta-base' or "distilroberta-base" (XLM-RoBERTa, Distil Roberta)
                                 # "google-bert/bert-base-cased" (mobileBert)

        random.seed(randnum)
        tf.random.set_seed(randnum)
        np.random.seed(randnum)
        
        model_output_name = "mbert_logging_test1"


        # Dataset split
        df = load_data('dataset/finaldataset_6k_shuffled_v2.csv')
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=randnum)
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=randnum)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df, tokenizer)

        # Model initialization
        model = TFAutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            # hidden_dropout_prob=0.3,
            # attention_probs_dropout_prob=0.15
        )
        
        # Freeze all layers
        for layer in model.layers:
            layer.trainable = False
        # Unfreeze classifier layer
        model.layers[-1].trainable = True
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"])
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
        # Prepare datasets
        train_ds = train_dataset.unbatch().batch(hp["batch_size"])
        val_ds = val_dataset.unbatch().batch(hp["batch_size"])

        # Prepare callbacks: EarlyStopping and ModelCheckpoint
        checkpoint_filepath = f"./checkpoints/{run_name}.h5"
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: [
                mlflow.log_metric("train_loss", logs["loss"], step=epoch),
                mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch),
                mlflow.log_metric("val_loss", logs["val_loss"], step=epoch),
                mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch),
            ])
        ]
        
        # Train the model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=hp["epochs"],
            callbacks=callbacks,
            verbose=1
        )
    
        # Optionally load best checkpoint
        if os.path.exists(checkpoint_filepath):
            model.load_weights(checkpoint_filepath)
        
        # Evaluate the model
        val_preds = model.predict(val_ds).logits
        y_pred = np.argmax(val_preds, axis=1)
        y_true = np.concatenate([y for x, y in val_ds], axis=0)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("||-------------------------------------------------------||")
        print(f"||--> Run {run_name} evaluation metrics output:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
        print("||-------------------------------------------------------||")
        # Log metrics to MLflow
        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1_score": f1
        })

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

        # Clean up GPU memory
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--exp_desc", type=str, required=False, default="oo")
    args = parser.parse_args()

    print(f"Learning Rate: {args.learning_rate}, Epochs: {args.epochs}, Batch Size: {args.batch_size} Model: {args.model} Exp Des: {args.exp_desc}")

    # Model Names: 
    # 'bert-base-uncased' (bert)
    # 'bert-base-multilingual-cased' (mBERT)
    # 'xlm-roberta-base' or "distilroberta-base" (XLM-RoBERTa, Distil Roberta)
    # "google-bert/bert-base-cased" (mobileBert)

    # Prepare hyperparameter dictionary
    hyperparams = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "exp_desc" : args.exp_desc
    }
    
    run_training(hyperparams, args.model)