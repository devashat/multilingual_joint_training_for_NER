import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from evaluate import load

# Paths to validation files
val_files = ["../validation/es_ES.jsonl", "../validation/fr_FR.jsonl"]

# Load tokenizer and fine-tuned model
model_checkpoint = "./t5_multilingual_translation_final"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load and combine multiple validation datasets
def load_val_datasets(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                json_obj = json.loads(line.strip())
                if json_obj.get("targets"):
                    data.append({
                        "source": json_obj["source"],
                        "target": json_obj["targets"][0]["translation"],
                        "translation_task": f"translate {json_obj['source_locale']} to {json_obj['target_locale']}: ",
                        "target_locale": json_obj["target_locale"]
                    })
    return Dataset.from_list(data)

print("\nLoading validation datasets...")
val_dataset = load_val_datasets(val_files)

# Translate all validation questions
def translate_questions(dataset):
    predictions = []
    for example in tqdm(dataset, desc="Translating", unit="example"):
        question = example["source"]
        prefix = example.get("translation_task", "translate English to Spanish: ")

        inputs = tokenizer(
            prefix + question,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs)

        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(translation)
    return predictions

# Compute BLEU score
metric = load("sacrebleu")

def compute_bleu(predictions, references):
    result = metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return result["score"]

print("\nTranslating combined validation dataset...")
val_predictions = translate_questions(val_dataset)
val_references = [ex["target"] for ex in val_dataset]

val_bleu = compute_bleu(val_predictions, val_references)
print(f"\nCombined Validation sacreBLEU Score: {val_bleu:.2f}")

# Compute per-language BLEU scores
def compute_bleu_per_language(dataset, predictions, references, locales=["es", "fr"]):
    results = {}
    for lang in locales:
        lang_indices = [i for i, ex in enumerate(dataset) if ex["target_locale"] == lang]
        if lang_indices:
            lang_preds = [predictions[i] for i in lang_indices]
            lang_refs = [references[i] for i in lang_indices]
            lang_bleu = metric.compute(predictions=lang_preds, references=[[ref] for ref in lang_refs])["score"]
            results[lang] = lang_bleu
        else:
            results[lang] = None
    return results

lang_bleu_scores = compute_bleu_per_language(val_dataset, val_predictions, val_references)

# Print per-language scores
for lang, bleu in lang_bleu_scores.items():
    if bleu is not None:
        print(f"{lang.upper()} sacreBLEU Score: {bleu:.2f}")
    else:
        print(f"{lang.upper()} sacreBLEU Score: No examples found for this language.")

# Save predictions and references
def save_predictions(predictions, references, output_file, dataset, bleu_score=None, lang_scores=None):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions):
            entry = {
                "source": dataset[i]["source"],
                "prediction": pred,
                "reference": references[i],
                "target_locale": dataset[i]["target_locale"]
            }
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
        if bleu_score is not None:
            f.write(json.dumps({"combined_sacreBLEU_score": bleu_score}) + "\n")
        if lang_scores:
            for lang, score in lang_scores.items():
                f.write(json.dumps({f"{lang}_sacreBLEU_score": score}) + "\n")

print("\nSaving combined validation predictions and BLEU scores...")
save_predictions(
    val_predictions,
    val_references,
    "./val_predictions_combined.jsonl",
    val_dataset,
    bleu_score=val_bleu,
    lang_scores=lang_bleu_scores
)
