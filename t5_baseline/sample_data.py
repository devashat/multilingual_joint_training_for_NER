import json
import datasets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
from evaluate import load
from tqdm import tqdm 

# Load data
file_path = "./sample/es_ES.jsonl"

data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading dataset", unit="line"):
        json_obj = json.loads(line.strip())
        if "targets" in json_obj and len(json_obj["targets"]) > 0:
            data.append({
                "source": json_obj["source"],  # English question
                "target": json_obj["targets"][0]["translation"]  # First Spanish translation
            })

print(f"Loaded {len(data)} examples.")

# Convert to HF dataset
dataset = Dataset.from_list(data)

# Split train/val/test
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Define tokenizer and model
model_checkpoint = "t5-base" # t5-small
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Preprocess
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

# training args
batch_size = 8
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_translation",
    evaluation_strategy="steps",
    eval_steps=100, 
    save_strategy="epoch",
    logging_strategy="steps", 
    logging_steps=50,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs"
)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# eval metric
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
print("\n Starting training...\n")
trainer.train()

# Save model
print("\n Saving trained model...")
trainer.save_model("./t5_translation")
tokenizer.save_pretrained("./t5_translation")
print("Model saved successfully.")

# Function to generate translations with tracking
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
    "Who played the lead role in the movie Torrente, the dumb arm of the law?",
    "What year was the movie Torrente, the dumb arm of the law released?"
]

print("\n Running example translations...\n")
for question in tqdm(example_questions, desc="Translating examples", unit="question"):
    translate_question(question)
