'''
HOW TO RUN:

python3 inference.py --ckpt_name=best_ckpt-9-epochs-joint-trained.tar --output_file=./output.txt --language=es

python3 inference.py --ckpt_name best_ckpt.tar --src_file ../data/val_source.txt --target_file ../data/val_target.txt --output_file ../data/base_translations.txt
python3 inference.py --ckpt_name finetuned_best_ckpt.tar --src_file ../data/test_es_src.txt --target_file ../data/test_es_trg.txt --output_file ../data/test_es_translations.txt
python3 inference.py --ckpt_name finetuned_best_ckpt.tar --src_file ../data/test_fr_src.txt --target_file ../data/test_fr_trg.txt --output_file ../data/test_fr_translations.txt

python3 inference.py --ckpt_name finetuned_best_ckpt.tar --src_file ../data/val_source.txt --target_file ../data/val_target.txt --output_file ../data/finetuned_translations.txt

'''


import torch
import os
import argparse
import nltk
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from constants import *
from custom_data import process_text
from transformer import Transformer
import sentencepiece as spm
import evaluate 
import re
import ast
import logging
import pytorch_lightning as pl

class Inference:
    def __init__(self, ckpt_name, max_length=200):
        self.max_length = max_length

        # Load SentencePiece tokenizer
        print("Loading SentencePiece tokenizer...")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{SP_DIR}/{joint_model_prefix}.model")

        # Load vocabulary
        self.src_i2w = {}
        self.trg_i2w = {}

        with open(f"{SP_DIR}/{joint_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word
            self.trg_i2w[i] = word  # Using same vocabulary

        print(f"Vocabulary size: {len(self.src_i2w)}")

        # Load Transformer model
        print("Loading Transformer model...")
        self.model = Transformer(vocab_size=len(self.src_i2w)).to(device)
        
        # Load checkpoint
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_model", ckpt_name))
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
        # Load COMET metric
        pl_logger = logging.getLogger("pytorch_lightning")
        pl_logger.setLevel(logging.ERROR)
        print("Loading COMET model...")
        self.comet = evaluate.load("comet")

        self.model.eval()

    def extract_mentions(self, sentence):
        """
        Naively extract capitalized phrases from a sentence as pseudo-entity mentions.
        """
        return re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', sentence)
    
    def get_mentions_from_references(self, data):
        mentions = {}
        for instance in data:
            instance_id = instance["id"]
            instance_mentions = set()
            for target in instance["targets"]:
                mention = target["mention"]
                instance_mentions.add(mention)
            mentions[instance_id] = instance_mentions
        return mentions

    def compute_entity_name_translation_accuracy(self, predictions, mentions, verbose=False):
        correct, total = 0, 0
        for instance_id, instance_mentions in mentions.items():
            assert instance_mentions, f"No mentions for instance {instance_id}"
            total += 1
            if instance_id not in predictions:
                if verbose:
                    print(f"No prediction for instance {instance_id}")
                continue
            prediction = predictions[instance_id].casefold()
            match_found = any(m.casefold() in prediction for m in instance_mentions)
            if match_found:
                correct += 1
            elif verbose:
                print(f"Prediction: {predictions[instance_id]}")
                print(f"Ground truth mentions: {instance_mentions}\n")
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def translate(self, src_text_list):
        print("Tokenizing and encoding input text...")
        tokenized_list = process_text(src_text_list, is_target=False)
        input_tensor = torch.LongTensor(tokenized_list).to(device)

        translations = []
        with torch.no_grad():
            for i in tqdm(range(len(input_tensor))):
                src_input = input_tensor[i].unsqueeze(0)  # Add batch dimension
                e_mask = (src_input != pad_id).unsqueeze(1)

                # Start decoding
                trg_input = torch.LongTensor([[sos_id]]).to(device)  # Start with <sos> # Change to <es> or <fr> depending on source input
                for _ in range(self.max_length):
                    d_mask = (trg_input != pad_id).unsqueeze(1)
                    nopeak_mask = torch.tril(torch.ones([1, trg_input.shape[1], trg_input.shape[1]], dtype=torch.bool)).to(device)
                    d_mask = d_mask & nopeak_mask

                    output = self.model(src_input, trg_input, e_mask, d_mask)
                    next_word = output.argmax(dim=-1)[:, -1].item()  # Get most probable next token

                    if next_word == eos_id:
                        break

                    trg_input = torch.cat((trg_input, torch.LongTensor([[next_word]]).to(device)), dim=1)

                # Convert tokenized output back to text
                translated_text = self.decode(trg_input.squeeze(0).tolist())
                translations.append(translated_text)

        return translations

    def decode(self, token_list):
        """Convert tokenized IDs back into readable text."""
        return self.sp.DecodeIds([tok for tok in token_list if tok not in [sos_id, eos_id, pad_id]])

    def infer_and_evaluate(self, src_file, target_file, output_file):
        # Load source sentences
        print(f"Loading source sentences from {src_file}...")
        with open(src_file, "r", encoding="utf-8") as f:
            src_text_list = [line.strip() for line in f.readlines() if line.strip()]

        # Load target sentences
        print(f"Loading target sentences from {target_file}...")
        
        # Parse each line as a list of strings (multiple references)
        target_text_list = []
        references_for_comet = []
        
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    refs = ast.literal_eval(line.strip())  # parse line like '["a", "b", "c"]'
                    tokenized_refs = [r.split() for r in refs]
                    target_text_list.append(tokenized_refs)
                    references_for_comet.append(refs)
                
        references = [refs[0] for refs in references_for_comet]  # COMET only supports one ref

        # Run inference
        translations = self.translate(src_text_list)

        # Compute BLEU score
        generated_tokens = [t.split() for t in translations]  # Tokenize generated text
        bleu_score = corpus_bleu(target_text_list, generated_tokens)  # Compute BLEU score

        # Compute COMET score with best match per reference
        print("Computing COMET score (best across references)...")

        best_comet_scores = []

        for i in tqdm(range(len(src_text_list))):
            src = src_text_list[i]
            hyp = translations[i]
            ref_list = references_for_comet[i]
            
            scores = []
            for ref in ref_list:
                comet_input = {
                    "sources": [src],
                    "predictions": [hyp],
                    "references": [ref],
                }
                result = self.comet.compute(**comet_input)
                scores.append(result["mean_score"])

            best_comet_scores.append(max(scores))

        comet_score = sum(best_comet_scores) / len(best_comet_scores)
        
        # Prepare data for M-ETA
        print("Preparing M-ETA inputs...")
        
        reference_data = []
        for i, ref_list in enumerate(references_for_comet):
            mentions = []
            for ref in ref_list:
                extracted = self.extract_mentions(ref)
                mentions.extend(extracted or [ref])  # fallback if empty
            reference_data.append({
                "id": f"Q{i}_0",
                "targets": [{"mention": m} for m in mentions]
            })
        
        mentions = self.get_mentions_from_references(reference_data)
        prediction_dict = {f"Q{i}_0": t for i, t in enumerate(translations)}

        # Compute M-ETA score
        print("Computing M-ETA score...")
        m_eta_result = self.compute_entity_name_translation_accuracy(
            prediction_dict, mentions, verbose=False 
        )
        
        meta_score = m_eta_result["accuracy"] * 100.0

        print(f"Saving translations, BLEU, and COMET scores to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for translation in translations:
                f.write(translation + "\n")
                
            f.write(f"BLEU Score: {bleu_score:.4f}\n")
            f.write(f"COMET Score: {comet_score:.4f}\n")
            f.write(f"M-ETA Score: {meta_score:.4f}\n")

        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"COMET Score: {comet_score:.4f}")
        print(f"M-ETA Score: {meta_score:.4f}")
        print("Inference complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", required=True, help="Checkpoint file to load (fine-tuned or non-fine-tuned model)")
    parser.add_argument("--src_file", required=False, help="Path to the source text file")
    parser.add_argument("--target_file", required=False, help="Path to the target text file for BLEU evaluation")
    parser.add_argument("--language", required=False, help="Language of the source text (es or fr)")
    parser.add_argument("--output_file", required=True, help="Path to save the generated translations and BLEU score")
    args = parser.parse_args()
    
    src_path = ""
    target_path = ""
    
    if args.src_file and args.target_file:
        src_path = args.src_file
        target_path = args.target_file
    elif args.language == "es":
        print("Using default Spanish test files...")
        src_path = "../data/es_test_src.txt"
        target_path = "../data/es_test_trg.txt"
    elif args.language == "fr":
        print("Using default French test files...")
        src_path = "../data/fr_test_src.txt"
        target_path = "../data/fr_test_trg.txt"
    else:
        raise ValueError("Invalid language specified. Use 'es' or 'fr'. or provide src_file and target_file")
        

    infer = Inference(ckpt_name=args.ckpt_name)
    infer.infer_and_evaluate(src_file=src_path, target_file=target_path, output_file=args.output_file)
