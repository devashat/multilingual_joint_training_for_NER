import json
import datasets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
from evaluate import load
from tqdm import tqdm 

# Load data
file_path = "./train/es/train.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading dataset", unit="line"):
        json_obj = json.loads(line.strip())
        data.append({
            "source": json_obj["source"],  # English question
            "target": json_obj["target"]   # Spanish translation
        })

print(f"Loaded {len(data)} examples.")

# Convert to HF dataset
dataset = Dataset.from_list(data)

# Split train/val/test
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Define tokenizer and model
model_checkpoint = "t5-base"  # Can change to "t5-small" for a lighter model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Preprocessing function
def preprocess_data(examples):
    inputs = ["translate English to Spanish: " + text for text in examples["source"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_data, batched=True, desc="Tokenizing")
print("Tokenization complete.")

# Training arguments
batch_size = 8
num_epochs = 5  # Run training for 5 epochs

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_translation",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save at every epoch
    save_total_limit=None,  # Keep all saved checkpoints (adjust as needed)
    logging_strategy="steps", 
    logging_steps=50,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_epochs,  # Train for 5 epochs
    predict_with_generate=True,
    logging_dir="./logs"
)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Evaluation metric
metric = load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": result["score"]}

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

# Train model
print("\nStarting training...\n")
trainer.train()

# Save final model
print("\nSaving trained model...")
trainer.save_model("./t5_translation_final")
tokenizer.save_pretrained("./t5_translation_final")
print("Final model saved successfully.")

# Function to generate translations
def translate_question(question):
    print(f"Translating: {question}")

    # Model on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inputs on cuda
    inputs = tokenizer("translate English to Spanish: " + question, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

    # Generate translation
    output = model.generate(**inputs)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Translation: {translation}")
    return translation

# Example inference
example_questions = [
    "Who played the lead role in the movie Titanic?",
    "What year was the first book of the A Song of Ice and Fire series published?"
]

print("\nRunning example translations...\n")
for question in tqdm(example_questions, desc="Translating examples", unit="question"):
    translate_question(question)
