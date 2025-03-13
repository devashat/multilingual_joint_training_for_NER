import torch

# Path or parameters for data
DATA_DIR = '../data'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'

# Updated raw data filenames for multi-task setup
SRC_RAW_DATA_NAME_FR = 'raw_data_fr.src'  # English source for French translation
TRG_RAW_DATA_NAME_FR = 'raw_data_fr.trg'  # French target

SRC_RAW_DATA_NAME_ES = 'raw_data_es.src'  # English source for Spanish translation
TRG_RAW_DATA_NAME_ES = 'raw_data_es.trg'  # Spanish target

JOINT_RAW_DATA_NAME = 'joint_multilingual_data.txt'  # Combined dataset for tokenizer training

TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
joint_model_prefix = 'joint_sp'  # NEW: Single tokenizer model for all languages

# Use a larger vocabulary to accommodate all three languages
sp_vocab_size = 32000  # Increased to handle both French and Spanish subwords
character_coverage = 1.0
model_type = 'unigram'

# Parameters for Transformer & training
device = torch.device('cuda')  # CHANGED
learning_rate = 1e-4
batch_size = 8
seq_len = 200
num_heads = 8
num_layers = 6
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 10
beam_size = 8
ckpt_dir = '../saved_model'
