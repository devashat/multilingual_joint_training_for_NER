from constants import *
import os
import sentencepiece as spm

train_frac = 0.8

def train_sp():
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

    joint_input_file = f"{DATA_DIR}/{JOINT_RAW_DATA_NAME}"
    model_prefix_path = f"{SP_DIR}/{joint_model_prefix}"

    config = template.format(joint_input_file,
                             pad_id,
                             sos_id,
                             eos_id,
                             unk_id,
                             model_prefix_path,
                             sp_vocab_size,
                             character_coverage,
                             model_type)

    print("Training joint SentencePiece tokenizer...")
    
    if not os.path.isdir(SP_DIR):
        os.mkdir(SP_DIR)

    spm.SentencePieceTrainer.Train(config)

def create_joint_dataset():
    """ Merges English-French and English-Spanish datasets into a single file for tokenizer training """
    with open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME_FR}") as src_fr_f, \
         open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME_FR}") as trg_fr_f, \
         open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME_ES}") as src_es_f, \
         open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME_ES}") as trg_es_f:

        src_fr_lines = src_fr_f.readlines()
        trg_fr_lines = trg_fr_f.readlines()
        src_es_lines = src_es_f.readlines()
        trg_es_lines = trg_es_f.readlines()

    with open(f"{DATA_DIR}/{JOINT_RAW_DATA_NAME}", "w") as out_f:
        for src, trg in zip(src_fr_lines, trg_fr_lines):
            out_f.write(f"{src.strip()}\n")
            out_f.write(f"<fr> {trg.strip()}\n")
        
        for src, trg in zip(src_es_lines, trg_es_lines):
            out_f.write(f"{src.strip()}\n")
            out_f.write(f"<es> {trg.strip()}\n")

    print("Merged dataset created at", f"{DATA_DIR}/{JOINT_RAW_DATA_NAME}")

def split_data():
    """ Splits the dataset into train and validation sets properly, ensuring alignment """
    os.makedirs(f"{DATA_DIR}/{SRC_DIR}", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/{TRG_DIR}", exist_ok=True)

    with open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME_FR}") as src_fr_f, \
         open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME_FR}") as trg_fr_f, \
         open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME_ES}") as src_es_f, \
         open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME_ES}") as trg_es_f:

        src_fr_lines = src_fr_f.readlines()
        trg_fr_lines = trg_fr_f.readlines()
        src_es_lines = src_es_f.readlines()
        trg_es_lines = trg_es_f.readlines()

    train_size_fr = int(len(src_fr_lines) * train_frac)
    train_size_es = int(len(src_es_lines) * train_frac)

    train_src_fr = src_fr_lines[:train_size_fr]
    train_trg_fr = trg_fr_lines[:train_size_fr]
    train_src_es = src_es_lines[:train_size_es]
    train_trg_es = trg_es_lines[:train_size_es]

    valid_src_fr = src_fr_lines[train_size_fr:]
    valid_trg_fr = trg_fr_lines[train_size_fr:]
    valid_src_es = src_es_lines[train_size_es:]
    valid_trg_es = trg_es_lines[train_size_es:]

    with open(f"{DATA_DIR}/{SRC_DIR}/{TRAIN_NAME}", "w") as train_src_f, \
         open(f"{DATA_DIR}/{TRG_DIR}/{TRAIN_NAME}", "w") as train_trg_f:
        for src, trg in zip(train_src_fr, train_trg_fr):
            train_src_f.write(src)
            train_trg_f.write(f"<fr> {trg}")
        for src, trg in zip(train_src_es, train_trg_es):
            train_src_f.write(src)
            train_trg_f.write(f"<es> {trg}")

    with open(f"{DATA_DIR}/{SRC_DIR}/{VALID_NAME}", "w") as valid_src_f, \
         open(f"{DATA_DIR}/{TRG_DIR}/{VALID_NAME}", "w") as valid_trg_f:
        for src, trg in zip(valid_src_fr, valid_trg_fr):
            valid_src_f.write(src)
            valid_trg_f.write(f"<fr> {trg}")
        for src, trg in zip(valid_src_es, valid_trg_es):
            valid_src_f.write(src)
            valid_trg_f.write(f"<es> {trg}")

    print(f"Train/Validation data saved in {DATA_DIR}/{SRC_DIR}/ and {DATA_DIR}/{TRG_DIR}/")

if __name__ == "__main__":
    create_joint_dataset()
    split_data()
    train_sp()
