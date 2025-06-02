import os
import shutil
import tempfile
import pickle as pkl
from pathlib import Path
from tqdm import tqdm
from typing import List

import torch
import pandas as pd

from esm.models.esm3 import ESM3
from esm.sdk.api import LogitsConfig, ESMProtein, GenerationConfig
from esm.utils.sampling import _BatchedESMProteinTensor
from CLEAN.model import LayerNormNet
from CLEAN.utils import get_ec_id_dict, seed_everything
from CLEAN.distance_map import get_dist_map_test
from CLEAN.evaluate import write_max_sep_choices, get_pred_labels, get_pred_probs

from oag.models import ESM1b_Embedding
from oag.constants import OUTPUT_FOLDER, CLEAN_MLP, CLEAN_DATA_FOLDER, CLEAN_RESULTS_TMP_FILE, CLEAN_GMM


def gen_samples_esm_and_clean_emb(sample_file: str, rebuild: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(sample_file, Path):
        sample_file = Path(sample_file)
    samples_dict = pkl.load(open(sample_file, "rb"))
    samples = samples_dict["sequences"]
    samples = [s.replace(" ", "") for s in samples]
    samples_dict["sequences"] = samples

    print(device)
    esm1b = ESM1b_Embedding()
    esm3 = ESM3.from_pretrained("esm3-sm-open-v1")
    esm1b.to(device)
    esm3.to(device)
    model = LayerNormNet(512, 128, device, torch.float32)
    checkpoint = torch.load(CLEAN_MLP, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    esm_emb_file = OUTPUT_FOLDER / f"{sample_file.stem}_esm_embeddings.pt"
    esm3_emb_file = OUTPUT_FOLDER / f"{sample_file.stem}_esm3_embeddings.pt"
    clean_emb_file = OUTPUT_FOLDER / f"{sample_file.stem}_clean_embeddings.pt"
    if not esm_emb_file.exists() or rebuild:
    # if not esm_emb_file.exists():
        esm_emb = get_esm1b_embeddings(esm1b, samples)
        torch.save(esm_emb, esm_emb_file)
    esm_emb = torch.load(esm_emb_file, map_location=device)

    if not esm3_emb_file.exists() or rebuild:
        esm3_emb = get_esm3_embeddings(esm3, samples)
        torch.save(esm3_emb, esm3_emb_file)

    esm3_emb = torch.load(esm3_emb_file, map_location=device)
    if not clean_emb_file.exists() or rebuild:
        clean_emb = []
        for i in range(0, len(esm_emb), 128):
            clean_emb.append(model(esm_emb[i:i + 128].to(device)))
        clean_emb = torch.cat(clean_emb, dim=0)
        torch.save(clean_emb, clean_emb_file)

    clean_emb = torch.load(clean_emb_file, map_location=device)

    return esm_emb, esm3_emb, clean_emb


from typing import List

_esm1b = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_clean_labels(samples: List[str]):
    global _esm1b, _model
    if _esm1b is None:
        _esm1b = ESM1b_Embedding()
        _esm1b.to(_device)
    if _model is None:
        _model = LayerNormNet(512, 128, _device, torch.float32)
        checkpoint = torch.load(CLEAN_MLP, map_location=_device)
        _model.load_state_dict(checkpoint)
        _model.eval()
        _model.to(_device)
    esm_emb = get_esm1b_embeddings(_esm1b, samples, verbose=False)
    clean_emb = _model(esm_emb.to(_device))
    pred_labels, pred_probs = clean_labels_probs_from_emb(clean_emb, verbose=False)
    return pred_labels, pred_probs


def clean_labels_probs_from_emb(clean_emb, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id_ec_train, ec_id_dict_train = get_ec_id_dict(CLEAN_DATA_FOLDER / "split100.csv")
    id_ec_test = {i: None for i in range(len(clean_emb))} # it's expecting a dict of ids to something
    emb_train = torch.load(CLEAN_DATA_FOLDER / 'pretrained/100.pt', map_location=device) # cluster embeddings, not model checkpoint

    # eval_dist = get_dist_map_test(emb_train, clean_emb, ec_id_dict_train, id_ec_test, device, torch.float32, verbose=verbose)
    eval_dist = get_dist_map_test(emb_train, clean_emb, ec_id_dict_train, id_ec_test, device, torch.float32)

    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    out_filename = str(CLEAN_RESULTS_TMP_FILE) # since this is originally a path
    write_max_sep_choices(eval_df, out_filename, gmm=CLEAN_GMM)

    pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
    pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
    return pred_label, pred_probs

from esm.sdk.api import ESMProtein, GenerationConfig
def get_esm3_structs(esm, new_pred_seqs):
    # Generate structure from the newly predicted sequences.
    struct_prompts = [ESMProtein(sequence=s) for s in new_pred_seqs]
    struct_configs = [
        GenerationConfig(
            track="structure",
            num_steps=max(len(new_pred_seqs[i]) // 8, 1),
            temperature=0.7
        )
        for i in range(len(new_pred_seqs))
    ]
    structs = esm.batch_generate(struct_prompts, struct_configs)
    # Move structure tensors to CPU.
    new_pred_structs = [struct.coordinates[:, :3].cpu() for struct in structs] # they take care of unpadding correctly for us
    new_plddt = [struct.plddt.cpu() for struct in structs]
    new_ptm = [struct.ptm.cpu() for struct in structs]
    return new_pred_structs, new_plddt, new_ptm


def get_esm1b_embeddings(model: ESM1b_Embedding, seqs: List[str], batch_size: int = 8, verbose=True) -> torch.Tensor:
    # should err on the side of smaller batches to reduce the compute spend on padding tokens
    # Create a temporary directory for caching batches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tmp_dir = tempfile.mkdtemp(prefix="esm1b_embeddings_cache_")
    embeddings = []
    chunk_files = []
    batch_count = 0

    for i in tqdm(range(0, len(seqs), batch_size), desc="Computing embeddings", disable=not verbose):
        batch = seqs[i:i+batch_size]
        with torch.no_grad():
            esm_embeddings, mask = model(batch)
            for embedding, m in zip(esm_embeddings, mask):
                idx = m.nonzero(as_tuple=True)[0]
                esm_embedding = embedding[idx].cpu()
                esm_embedding = esm_embedding[1:len(embedding) - int(model.has_eos)].mean(dim=0)
                embeddings.append(esm_embedding)
        batch_count += 1

        if batch_count % 100 == 0:
            # Save the current batch embeddings to a temporary file
            chunk_file = os.path.join(tmp_dir, f"chunk_{batch_count//100}.pt")
            torch.save(torch.stack(embeddings).cpu(), chunk_file)
            chunk_files.append(chunk_file)
            embeddings = []  # reset for next batches

    # Save any remaining embeddings that haven't been cached yet
    if embeddings:
        chunk_file = os.path.join(tmp_dir, f"chunk_{(batch_count // 100) + 1}.pt")
        torch.save(torch.stack(embeddings), chunk_file)
        chunk_files.append(chunk_file)

    # Load all chunk files in order and concatenate them into one tensor
    all_embeddings = []
    for file in sorted(chunk_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        all_embeddings.append(torch.load(file))

    try:
        final_embeddings = torch.cat(all_embeddings, dim=0)
    except RuntimeError as e:
        print(f"Error concatenating embeddings: {e}")
        raise

    # Delete the temporary directory after successful concatenation
    shutil.rmtree(tmp_dir)

    return final_embeddings

def get_esm3_embeddings(model: ESM3, seqs: List[str], batch_size: int = 4) -> torch.Tensor:
    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    tmp_dir = tempfile.mkdtemp(prefix="esm3_embeddings_cache_")
    chunk_files = []
    cache_index = 0
    batch_counter = 0
    embeddings_LRE = []
    for batch_st in tqdm(range(0, len(seqs), batch_size), desc="Shard progress"):
        batch_counter += 1
        batch_seqs = seqs[batch_st:batch_st+batch_size]
        tokenized_seqs = [model.tokenizers.sequence.encode(seq) for seq in batch_seqs]
        max_len = max(len(seq) for seq in tokenized_seqs)
        padded_seqs_BR = [seq + [1] * (max_len - len(seq)) for seq in tokenized_seqs]  # 1 is the padding token for ESM3
        padded_seqs_BR = torch.tensor(padded_seqs_BR).cuda()
        padded_seqs_BR = _BatchedESMProteinTensor(sequence=padded_seqs_BR)
        with torch.no_grad():
            logits = model.logits(padded_seqs_BR, logits_config)
            for tokenized_seq, embedding_RE in zip(tokenized_seqs, logits.embeddings):
                embedding_RE = embedding_RE[:len(tokenized_seq)]
                embeddings_LRE.append(embedding_RE.mean(dim=0).cpu())
        if batch_counter % 200 == 0:
            cache_file = os.path.join(tmp_dir, f"chunk_{cache_index}.pt")
            torch.save(torch.stack(embeddings_LRE), cache_file)
            chunk_files.append(cache_file)
            cache_index += 1
            embeddings_LRE = []


    if len(embeddings_LRE) > 0:
        cache_file = os.path.join(tmp_dir, f"chunk_{cache_index}.pt")
        torch.save(torch.stack(embeddings_LRE).cpu(), cache_file)
        chunk_files.append(cache_file)
        cache_index += 1
        embeddings_LRE = []  # Clear the current embeddings

    # Load all cached embeddings and concatenate them in order
    # for file in sorted(chunk_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
    #     all_embeddings.append(torch.load(file))
    all_embeddings = [torch.load(file).cpu() for file in chunk_files]
    final_embeddings = torch.cat(all_embeddings, dim=0) # since we stacked them before saving

    shutil.rmtree(tmp_dir)
    return final_embeddings

def get_esm3_structs(esm, new_pred_seqs):
    # Generate structure from the newly predicted sequences.
    struct_prompts = [ESMProtein(sequence=s) for s in new_pred_seqs]
    struct_configs = [
        GenerationConfig(
            track="structure",
            num_steps=max(len(new_pred_seqs[i]) // 8, 1),
            temperature=0.7
        )
        for i in range(len(new_pred_seqs))
    ]
    structs = esm.batch_generate(struct_prompts, struct_configs)
    # Move structure tensors to CPU.
    new_pred_structs = [struct.coordinates[:, :3].cpu() for struct in structs] # they take care of unpadding correctly for us
    new_plddt = [struct.plddt.cpu() for struct in structs]
    new_ptm = [struct.ptm.cpu() for struct in structs]
    return new_pred_structs, new_plddt, new_ptm
