import os
import pickle as pkl
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

HOME_FOLDER = Path(__file__).parent.parent.parent
DATA_FOLDER = HOME_FOLDER / "data"
DB_FOLDER = HOME_FOLDER / "db"
OUTPUT_FOLDER = HOME_FOLDER / "output"

CHECKPOINT_FOLDER = DB_FOLDER / "checkpoints"

class ModelType(Enum):
    ESM1 = "esm1b"
    ESM3 = "esm3"

# DON'T CHANGE SET MANUALLY WHILE RUNNING DATASET CREATION WITH DATA.PY CODE
# ALTHOUGH SOME OF THESE MIGHT NO LONGER BE USED IN THE CURRENT CODE
EC_DATASET_FOLDER = Path(os.environ["EC_DATASET_FOLDER"])
SWISSPROT_FASTA = EC_DATASET_FOLDER / "uniprot_sprot.fasta"
SWISSPROT_RECORDS_FOLDER = EC_DATASET_FOLDER / "proteinfer_dataset"
SWISSPROT_EC_FOLDER = EC_DATASET_FOLDER / "swiss_prot_ec/" # contains the transformed samples from the proteinfer dataset as json, along with the versions with multi-hot torch.tensor labels, saved as pkl
SWISSPROT_MULTIHOT_FOLDER = EC_DATASET_FOLDER / "swiss_prot_ec_multi_hot"
SWISSPROT_EMBED_ESM1_FOLDER = EC_DATASET_FOLDER / "swiss_prot_ec_embeddings_esm1b"
SWISSPROT_EMBED_ESM3_FOLDER = EC_DATASET_FOLDER / "swiss_prot_ec_embeddings_esm3"
SWISSPROT_UNMASKED_DATASET_FOLDER = EC_DATASET_FOLDER / "swiss_prot_ec_unmasked_dataset" # contains the cache files for different splits for the unmasked dataset
SWISSPROT_MASKED_DATASET_FOLDER = EC_DATASET_FOLDER / "swiss_prot_ec_masked_dataset" # contains the cache files for different splits for the masked dataset

LEVEL_SETS_FILE = SWISSPROT_EC_FOLDER / "level_ec_to_idx.pkl"
IDX_TO_EC_LEVEL = pkl.load(open(LEVEL_SETS_FILE, "rb"))

for folder in [
    SWISSPROT_RECORDS_FOLDER,
    SWISSPROT_EC_FOLDER,
    SWISSPROT_MULTIHOT_FOLDER,
    SWISSPROT_EMBED_ESM1_FOLDER,
    SWISSPROT_EMBED_ESM3_FOLDER,
    SWISSPROT_UNMASKED_DATASET_FOLDER,
    SWISSPROT_MASKED_DATASET_FOLDER
]:
    if not folder.exists():
        folder.mkdir()

CLEAN_FOLDER = Path(os.environ["CLEAN_FOLDER"])
CLEAN_DATA_FOLDER = CLEAN_FOLDER / "data"
CLEAN_CHECKPOINT_FOLDER = CLEAN_DATA_FOLDER / "pretrained"
CLEAN_MLP = CLEAN_CHECKPOINT_FOLDER / "split100.pth"
CLEAN_RESULTS_TMP_FILE = CLEAN_FOLDER / "results"
CLEAN_GMM = CLEAN_CHECKPOINT_FOLDER / "gmm_ensumble.pkl"

UNMASKED_DATASET_EMBEDDINGS = SWISSPROT_UNMASKED_DATASET_FOLDER / "clean_embeddings.pt"
UNMASKED_DATASET_LABELS = {f"level_{i}": SWISSPROT_UNMASKED_DATASET_FOLDER / f"labels_level_{i}.pt" for i in range(1, 5)}
UNMASKED_DATASET_PIDS = SWISSPROT_UNMASKED_DATASET_FOLDER / "protein_ids.pkl"

class ModelType(Enum):
    ESM1 = "esm1b"
    ESM3 = "esm3"

CHECKPOINT_FOLDER = Path("/home/ishan/protein_discrete_guidance/pdg/esm_ec/checkpoints")
CLEAN_CLUSTER_CENTERS = CHECKPOINT_FOLDER / "cluster_centers.pt"