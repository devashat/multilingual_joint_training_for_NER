import json
import os
import datasets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
from evaluate import load
from tqdm import tqdm

data_dir = "../train"
languages = ["es", "fr"]  # trying with two languages

def load_multilingual_data(data_dir, languages):
    data = []
    for language in languages:
        file_path = os.path.join(data_dir, language, "train.jsonl")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Loading dataset {language}", unit="line"):
                    json_obj = json.loads(line.strip())
                    data.append({
                        "source": json_obj["source"],
                        "target": json_obj["target"],
                        "translation_task": f"translate {json_obj['source_locale']} to {json_obj['target_locale']}: "
                    })
    return data

data = load_multilingual_data(data_dir, languages)
print(f"Loaded {len(data)} examples from Spanish and French.")

dataset = Dataset.from_list(data)

dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset["train"].train_test_split(test_size=0.1)

model_checkpoint = "t5-base"  # "t5-small" for a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_data(examples):
    inputs = [task + text for task, text in zip(examples["translation_task"], examples["source"])]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_data, batched=True, desc="Tokenizing")
print("Tokenization complete.")

batch_size = 8
num_epochs = 5

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_multilingual_translation",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=None,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    logging_dir="./logs"
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

metric = load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": result["score"]}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

print("\nStarting training...\n")
trainer.train()

print("\nSaving trained model...")
trainer.save_model("./t5_multilingual_translation_final")
tokenizer.save_pretrained("./t5_multilingual_translation_final")
print("Final model saved successfully.")

def translate_question(question, src_lang="en", tgt_lang="es"):
    print(f"Translating: {question}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(f"translate {src_lang} to {tgt_lang}: " + question, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(**inputs)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Translation: {translation}")
    return translation

example_questions = [
    ("Who played the lead role in the movie Titanic?", "en", "es"),
    ("What year was the first book of the A Song of Ice and Fire series published?", "en", "fr")
]

print("\nRunning example translations...\n")
for question, src_lang, tgt_lang in tqdm(example_questions, desc="Translating examples", unit="question"):
    translate_question(question, src_lang, tgt_lang)