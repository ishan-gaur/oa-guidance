# import torch
# import numpy as np
# import pickle as pkl
# from tqdm import tqdm
# from pathlib import Path
# from esm.models.esm3 import ESM3
# from pdg.esm_ec.src.model import ESM1b_Embedding, SingleClassifier
# from pdg.esm_ec.constants import DATA_FOLDER, SWISSPROT_EC_FOLDER, LEVEL_SETS, SWISSPROT_UNMASKED_DATASET_FOLDER
# from pdg.esm_ec.src.data import ECEmbeddingDataset, ModelType
# from pdg.esm_ec.src.guidance import esm3_class_guide_seq
# from pdg.esm_ec.src.eval import get_esm3_structs
# from pdg.esm_ec.src.utils import mask, log_file_path
# from pdg.esm_ec.src.guidance import STEPS_CONFIG
from oag.logging import log_file_path, Logger

def main(
    mask_level = 1.0,
    guidance_temp = 1.0,
    x1_temp = 1.0, # ESM Default is actually 0.7
    use_tag = False,
    num_classes = 10,
):
    # step_strategy = STEPS_CONFIG.GILLESPIE
    level_str = "level_4"

    data_file = log_file_path(
        experiment_name="traces_for_probing_divergence",
        mask_level=mask_level,
        guidance_temp=guidance_temp,
        x1_temp=x1_temp,
        use_tag=use_tag,
    )
    print(f"Data file: {data_file}")
    logger = Logger(fields=["target_ec_str"])


    # get the target ECs
    LEVEL_NUM = int(level_str.split("_")[-1])
    get_level_ec = lambda x: ".".join(x.split(".")[:LEVEL_NUM])
    NON_ENZ_EC = get_level_ec("EC:0.0.0.0")
    try:
        logger.assert_fields_logged()
    except ValueError as e:
        print(f"Error: {e}")
    try:
        logger.assert_samples_logged([0])
    except ValueError as e:
        print(f"Error: {e}")
    logger.log("target_ec_str", NON_ENZ_EC, 0)
    logger.assert_fields_logged()
    logger.assert_samples_logged([0])
    try:
        logger.assert_samples_logged([1])
    except ValueError as e:
        print(f"Error: {e}")
    print(logger)
    sample = logger.get_sample(0)
    print(sample["target_ec_str"][0])
    logger.to_file(data_file)
    logger = Logger.from_file(data_file)
    print(logger)

    # level_sets = pkl.load(open(SWISSPROT_EC_FOLDER / LEVEL_SETS, "rb"))
    # idx_to_ec = level_sets[level_str]
    # ec_to_idx = {get_level_ec(ec): i for i, ec in idx_to_ec.items()}
    # label_mask = list(range(len(level_sets["level_4"])))

#     test_dataset = ECEmbeddingDataset.from_file(SWISSPROT_UNMASKED_DATASET_FOLDER, split="test", model=ModelType.ESM3)
#     if top_ecs:
#         target_ecs = pkl.load(open(SWISSPROT_EC_FOLDER / "top_k_ecs.pkl", "rb"))
#         target_ecs = [get_level_ec(ec) for ec in target_ecs]
#         target_ecs = target_ecs[:1 + num_classes] # 1 is for the non-enzymatic class
#         target_ecs = list(set(target_ecs) - set([NON_ENZ_EC]))
#     else:
#         non_neg_rows = torch.nonzero(test_dataset.labels_4.sum(dim=1) > 0).squeeze(1)
#         test_labels = torch.nonzero(torch.sum(test_dataset.labels_4[non_neg_rows], dim=0) > 0).squeeze(1)
#         test_ecs = [get_level_ec(idx_to_ec[i]) for i in test_labels.tolist()]
#         target_ecs = sorted(list(set(test_ecs) - {NON_ENZ_EC}))
#         np.random.seed(42)  # Using 42 as a standard seed value
#         # Randomly select 10 target ECs from the available ECs
#         target_ecs = np.random.choice(target_ecs, size=min(10, len(target_ecs)), replace=False).tolist()

#     print(f"Number of total samples: {num_per_class * len(target_ecs)}")

#     # keys for the saved data file
#     keys = ["sequences", "structures", "plddt_mean", "plddt_median", "plddt_min", "plddt_max", "plddt_std", "ptm", "target_ec", "paths", "logits_px", "logits_pyx"]
#     struct_keys = ["structures", "plddt_mean", "plddt_median", "plddt_min", "plddt_max", "plddt_std", "ptm"]


#     device = "cuda"
#     esm = ESM3.from_pretrained("esm3-sm-open-v1").to(device)
#     esm.to(dtype=torch.float32)
#     esm.to(device)
#     classifier = SingleClassifier(classifier_hash="mrdh1641", levels_dict=level_sets, label_mask=label_mask, time_disc=2.0)
#     classifier.to(device)
#     for ec in tqdm(target_ecs, desc="Target EC progress"):
#         # sample masked sequences
#         # set model sampling function and log prob function to match target class
#         # go through batches to get the samples
#         # save the samples, and structure metrics, along with the target EC
#         idx = ec_to_idx[ec]
#         if level_str == "level_1":
#             labels = test_dataset.labels_1
#         elif level_str == "level_2":
#             labels = test_dataset.labels_2
#         elif level_str == "level_3":
#             labels = test_dataset.labels_3
#         elif level_str == "level_4":
#             labels = test_dataset.labels_4
#         # pick out the indices of samples that meet the target class
#         indices = torch.nonzero(labels[:, idx] == 1).squeeze(1)
#         sequence_indices = np.random.choice(indices, size=min(num_per_class, len(indices)), replace=False)
#         if len(sequence_indices) < num_per_class:
#             print(f"Not enough sequences for {ec}: {len(sequence_indices)} < {num_per_class}, continuing with {len(sequence_indices)}")
#         # don't split the dataset and just pull the sequences
#         sequences = [test_dataset.sequences[i] for i in sequence_indices]
#         mask_num = [int(len(s) * mask_level) for s in sequences]
#         masked_seqs = [mask(s, mask_num=m) for s, m in zip(sequences, mask_num)]

#         for batch_st in range(0, len(masked_seqs), batch_size):
#             batch_seqs = masked_seqs[batch_st:batch_st+batch_size]
#             samples = esm3_class_guide_seq(
#                 esm, classifier, target_class=(LEVEL_NUM - 1, idx), masked_sequences=batch_seqs, progress_bar=True,
#                 guide_temp=guidance_temp, use_tag=use_tag, batch_size=batch_size, x1_temp=x1_temp,
#                 stochasticity=stochasticity, num_classes=len(idx_to_ec), steps_strategy=step_strategy,
#                 verbose=True, unconditional=False, data_guide=True
#             )
#             sampled_seqs, sampled_paths, logits_px, logits_pyx = samples
#             if any([len(s) > ESM1b_Embedding.ESM_MAX_LEN for s in sampled_seqs]):
#                 print(f"Sampled sequence length exceeds max length: {ESM1b_Embedding.ESM_MAX_LEN}")
#                 continue

#             sampled_structs, plddt, ptm = get_esm3_structs(esm, sampled_seqs)

#             # Load existing data (if any) and append this batch.
#             if data_file.exists():
#                 with open(data_file, "rb") as f:
#                     saved = pkl.load(f)
#                 if not all(k in saved for k in struct_keys):
#                     for k in struct_keys:
#                         if k in saved: # just regenerate all for simplicity
#                             del saved[k]
#                         saved[k] = []
#                     for i in tqdm(range(0, len(saved["sequences"]), batch_size), desc="Regenerating structures"):
#                         prev_struct, prev_plddt, prev_ptm = get_esm3_structs(esm, saved["sequences"][i:i+batch_size])
#                         saved["structures"].extend(prev_struct)
#                         saved["plddt_mean"].extend([plddt.mean().item() for plddt in prev_plddt])
#                         saved["plddt_median"].extend([plddt.median().item() for plddt in prev_plddt])
#                         saved["plddt_min"].extend([plddt.min().item() for plddt in prev_plddt])
#                         saved["plddt_max"].extend([plddt.max().item() for plddt in prev_plddt])
#                         saved["plddt_std"].extend([plddt.std().item() for plddt in prev_plddt])
#                         saved["ptm"].extend([p.item() for p in prev_ptm])
#                 if not all(k in saved for k in keys):
#                     print(f"Missing keys in {data_file}: {keys}")
#                     print("Padding with None")
#                     for k in keys:
#                         if k not in saved:
#                             saved[k] = [None] * len(saved["sequences"])
#                 with open(data_file, "wb") as f:
#                     pkl.dump(saved, f)
#             else:
#                 keys = ["sequences", "structures", "plddt_mean", "plddt_median", "plddt_min", "plddt_max", "plddt_std", "ptm", "target_ec", "paths", "logits_px", "logits_pyx"]
#                 saved = {k: [] for k in keys}

#             saved["sequences"].extend(sampled_seqs)
#             saved["structures"].extend(sampled_structs)
#             saved["plddt_mean"].extend([plddt.mean().item() for plddt in plddt])
#             saved["plddt_median"].extend([plddt.median().item() for plddt in plddt])
#             saved["plddt_min"].extend([plddt.min().item() for plddt in plddt])
#             saved["plddt_max"].extend([plddt.max().item() for plddt in plddt])
#             saved["plddt_std"].extend([plddt.std().item() for plddt in plddt])
#             saved["ptm"].extend([p.item() for p in ptm])
#             saved["target_ec"].extend([ec] * len(sampled_seqs))
#             saved["paths"].extend(sampled_paths)
#             saved["logits_px"].extend(logits_px)
#             saved["logits_pyx"].extend(logits_pyx)

#             raise NotImplementedError("Saving data is not implemented yet.")
#             # with open(data_file, "wb") as f:
#             #     pkl.dump(saved, f)

if __name__ == "__main__":
    import fire
    fire.Fire(main)