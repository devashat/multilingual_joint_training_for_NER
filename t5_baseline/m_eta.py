import json
import re
from typing import Dict, Set

def extract_mentions(references) -> Dict[str, Set[str]]:
    mentions = {}
    for ex in references:
        instance_id = re.match(r"Q[0-9]+_[0-9]", ex["id"]).group(0)
        mentions[instance_id] = set(m["mention"] for m in ex["targets"] if "mention" in m)
    return mentions

def compute_m_eta(predictions: Dict[str, str], references: list, verbose: bool = False) -> float:
    mentions = extract_mentions(references)
    correct, total = 0, 0

    for instance_id, instance_mentions in mentions.items():
        total += 1
        pred = predictions.get(instance_id, "")
        pred = pred.casefold()

        entity_match = any(mention.casefold() in pred for mention in instance_mentions)
        if entity_match:
            correct += 1
        elif verbose:
            print(f"❌ Missed entity for {instance_id}\nPrediction: {pred}\nMentions: {instance_mentions}\n")

    return correct / total if total > 0 else 0.0
