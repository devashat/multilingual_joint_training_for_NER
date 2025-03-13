import os

os.makedirs("./test/src", exist_ok=True)
os.makedirs("./test/trg", exist_ok=True)

DATA_DIR = "."
SRC_RAW_DATA_NAME_FR = "raw_data_fr.src"
TRG_RAW_DATA_NAME_FR = "raw_data_fr.trg"
SRC_RAW_DATA_NAME_ES = "raw_data_es.src"
TRG_RAW_DATA_NAME_ES = "raw_data_es.trg"

src_fr_f = open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME_FR}")
trg_fr_f = open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME_FR}")
src_es_f = open(f"{DATA_DIR}/{SRC_RAW_DATA_NAME_ES}")
trg_es_f = open(f"{DATA_DIR}/{TRG_RAW_DATA_NAME_ES}")

src_fr_lines = src_fr_f.readlines()
trg_fr_lines = trg_fr_f.readlines()
src_es_lines = src_es_f.readlines()
trg_es_lines = trg_es_f.readlines()

train_percent = 0.8
train_size_fr = int(len(src_fr_lines) * train_percent)
train_size_es = int(len(src_es_lines) * train_percent)

train_src_fr_lines = src_fr_lines[:train_size_fr]
train_trg_fr_lines = trg_fr_lines[:train_size_fr]
train_src_es_lines = src_es_lines[:train_size_es]
train_trg_es_lines = trg_es_lines[:train_size_es]

valid_src_fr_lines = src_fr_lines[train_size_fr:]
valid_trg_fr_lines = trg_fr_lines[train_size_fr:]
valid_src_es_lines = src_es_lines[train_size_es:]
valid_trg_es_lines = trg_es_lines[train_size_es:]

# Train fr
for i in range(len(train_src_fr_lines)):
    source = train_src_fr_lines[i]
    target = "<fr> " + train_trg_fr_lines[i]
    with open("./test/src/train.txt", "a") as f:
        f.write(source)
        
    with open("./test/trg/train.txt", "a") as f:
        f.write(target)

# Train es
for i in range(len(train_src_es_lines)):
    source = train_src_es_lines[i]
    target = "<es> " + train_trg_es_lines[i]
    
    with open("./test/src/train.txt", "a") as f:
        f.write(source)
        
    with open("./test/trg/train.txt", "a") as f:
        f.write(target)

# Valid fr
for i in range(len(valid_src_fr_lines)):
    source = valid_src_fr_lines[i]
    target = "<fr> " + valid_trg_fr_lines[i]
    
    with open("./test/src/valid.txt", "a") as f:
        f.write(source)
        
    with open("./test/trg/valid.txt", "a") as f:
        f.write(target)
        
# Valid es
for i in range(len(valid_src_es_lines)):
    source = valid_src_es_lines[i]
    target = "<es> " + valid_trg_es_lines[i]
    
    with open("./test/src/valid.txt", "a") as f:
        f.write(source)
        
    with open("./test/trg/valid.txt", "a") as f:
        f.write(target)

src_fr_f.close()
trg_fr_f.close()
src_es_f.close()
trg_es_f.close()
    