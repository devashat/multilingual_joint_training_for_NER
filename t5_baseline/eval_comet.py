import json
import os
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from evaluate import load
from m_eta import compute_m_eta

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
                    mentions = list({t["mention"] for t in json_obj["targets"] if "mention" in t})
                    data.append({
                        "source": json_obj["source"],
                        "target": json_obj["targets"][0]["translation"],  # first reference for BLEU
                        "translation_task": f"translate {json_obj['source_locale']} to {json_obj['target_locale']}: ",
                        "target_locale": json_obj["target_locale"],
                        "entity_mentions": mentions
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

# Compute BLEU, COMET, M-ETA, Harmonic Mean
metric_bleu = load("sacrebleu")
metric_comet = load("comet")

def compute_all_metrics(predictions, references, val_dataset):
    # BLEU (overall)
    bleu_result = metric_bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    bleu_score = bleu_result["score"]

    # Sources for COMET
    sources = val_dataset["source"]

    # COMET
    comet_result = metric_comet.compute(sources=sources, predictions=predictions, references=references)
    if isinstance(comet_result, list):
        comet_score = sum([r["score"] for r in comet_result]) / len(comet_result)
    elif isinstance(comet_result, dict) and "score" in comet_result:
        comet_score = comet_result["score"]
    else:
        comet_score = 0.0

    # M-ETA (overall)
    references_meta = []
    predictions_meta = {}
    for i, mentions in enumerate(val_dataset["entity_mentions"]):
        instance_id = f"Q{i}_0"
        references_meta.append({
            "id": instance_id,
            "targets": [{"mention": m} for m in mentions]
        })
        predictions_meta[instance_id] = predictions[i]
    m_eta_score = compute_m_eta(predictions_meta, references_meta)

    # Harmonic mean (overall)
    if comet_score + m_eta_score > 0:
        harmonic_mean = 2 * (comet_score * m_eta_score) / (comet_score + m_eta_score)
    else:
        harmonic_mean = 0.0

    # ---------- Language-specific metrics ----------
    lang_metrics = {}
    for lang in ["es", "fr"]:
        lang_indices = [i for i, ex in enumerate(val_dataset) if ex["target_locale"] == lang]
        if not lang_indices:
            continue
        lang_preds = [predictions[i] for i in lang_indices]
        lang_refs = [references[i] for i in lang_indices]
        lang_sources = [val_dataset[i]["source"] for i in lang_indices]

        # COMET for this language
        lang_comet_result = metric_comet.compute(sources=lang_sources, predictions=lang_preds, references=lang_refs)

        # Handle format variations in COMET result
        if isinstance(lang_comet_result, list):
            if isinstance(lang_comet_result[0], dict) and "score" in lang_comet_result[0]:
                lang_comet = sum([r["score"] for r in lang_comet_result]) / len(lang_comet_result)
            elif isinstance(lang_comet_result[0], (float, int)):
                lang_comet = sum(lang_comet_result) / len(lang_comet_result)
            else:
                lang_comet = 0.0
        elif isinstance(lang_comet_result, dict) and "score" in lang_comet_result:
            lang_comet = lang_comet_result["score"]
        else:
            lang_comet = 0.0


        # M-ETA for this language
        lang_refs_meta = []
        lang_preds_meta = {}
        for idx, i in enumerate(lang_indices):
            instance_id = f"Q{i}_0"
            lang_refs_meta.append({
                "id": instance_id,
                "targets": [{"mention": m} for m in val_dataset[i]["entity_mentions"]]
            })
            lang_preds_meta[instance_id] = predictions[i]
        lang_m_eta = compute_m_eta(lang_preds_meta, lang_refs_meta)

        # Harmonic mean
        if lang_comet + lang_m_eta > 0:
            lang_hmean = 2 * (lang_comet * lang_m_eta) / (lang_comet + lang_m_eta)
        else:
            lang_hmean = 0.0

        lang_metrics[f"{lang}_comet"] = lang_comet
        lang_metrics[f"{lang}_m_eta"] = lang_m_eta
        lang_metrics[f"{lang}_harmonic_mean"] = lang_hmean

    # Return full metrics dictionary
    return {
        "bleu": bleu_score,
        "comet": comet_score,
        "m_eta": m_eta_score,
        "harmonic_mean_comet_meta": harmonic_mean,
        **lang_metrics
    }




# Translate and evaluate validation set
print("\nTranslating validation dataset...")
val_predictions = translate_questions(val_dataset)
val_references = [ex["target"] for ex in val_dataset]

metrics = compute_all_metrics(val_predictions, val_references, val_dataset)

# Print metrics
print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save predictions and metrics
def save_predictions(predictions, references, dataset, output_file, metrics_dict=None):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions):
            entry = {
                "source": dataset[i]["source"],
                "prediction": pred,
                "reference": references[i],
                "target_locale": dataset[i]["target_locale"],
                "entity_mentions": dataset[i]["entity_mentions"]
            }
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
        if metrics_dict:
            for key, val in metrics_dict.items():
                f.write(json.dumps({key: val}) + "\n")

print("\nSaving combined predictions and metrics...")
save_predictions(
    val_predictions,
    val_references,
    val_dataset,
    "./val_predictions_combined_with_metrics.jsonl",
    metrics_dict=metrics
)
