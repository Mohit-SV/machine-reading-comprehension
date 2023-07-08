import os

# Paths needed for creation of Visual SQuAD dataset
SRC_DIRECTORY = os.path.dirname(__file__)
TRAINSET_PATH = os.path.join(SRC_DIRECTORY, "data", "squad", "train-v2.0.json")
DEVSET_PATH = os.path.join(SRC_DIRECTORY, "data", "squad", "dev-v2.0.json")
V_SQUAD_DIR = os.path.join(SRC_DIRECTORY, "outputs", "squad", "training_pr_data")
DOC_LIMIT = None
ZIP_PATH = os.path.join(SRC_DIRECTORY, "outputs", "squad", "training_pr_data.zip")
INCLUDE_REFERENCES = False # not used yet

# Models
LLMv3_BACKBONE = "microsoft/layoutlmv3-base"
RoBERTa_BACKBONE = "roberta-base"

# Domain adaptaion
N_DOMAINS = 5 # number of domains into which dataset is to be split
N_TRAIN_DOMAINS, N_DEV_DOMAINS, N_TEST_DOMAINS = 3, 1, 1
