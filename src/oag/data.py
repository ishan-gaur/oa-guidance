import json
import torch
import copy
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Set, cast, Optional
from torch.utils.data import Dataset
from esm.models.esm3 import ESM3
from oag.eval import clean_labels_probs_from_emb, get_esm1b_embeddings, get_esm3_embeddings
from oag.constants import (
    SWISSPROT_EC_FOLDER, CLEAN_DATA_FOLDER, IDX_TO_EC_LEVEL, CLEAN_MLP, ModelType
)
from oag.models import ESM1b_Embedding
from oag.utils import mask as uniformly_mask
from CLEAN.utils import get_ec_id_dict
from CLEAN.model import LayerNormNet


def iterate_shards(dataset_folder: Path, with_labels: bool = False, desc: str = "Iterating through shards"):
    for shard_file in tqdm(list(dataset_folder.glob("*.json")), desc=desc):
        if not with_labels:
            with open(shard_file, "r") as f:
                examples = json.load(f)
            for example in examples:
                yield example
        else:
            with open(shard_file.with_suffix(".pkl"), "rb") as f:
                examples = pkl.load(f)
            for example in examples:
                yield example


class ECEmbeddingDataset(Dataset):
    def __init__(self,
        embeddings: torch.Tensor,
        labels: List[torch.Tensor],
        ids: List[str],
        sequences: List[str],
        omit_non_enzymes: bool = False,
        force_all_levels: bool = False,
        clean_labels_only: bool = False,
        include_mask_level: bool = False
    ):
        self.embeddings = embeddings
        assert len(labels) == 4
        self.labels_1, self.labels_2, self.labels_3, self.labels_4 = labels
        self.ids = ids
        self.sequences = sequences

        print("Initial samples:", len(self.ids))
        if omit_non_enzymes:
            self.filter_out_non_enzymes()
        if force_all_levels:
            self.filter_out_missing_levels()
        if clean_labels_only:
            self.filter_out_non_clean_labels()

        self.include_mask_level = include_mask_level
        if include_mask_level:
            self.mask_level = [
                self.sequences[i].count("<mask>") / (
                    self.sequences[i].count("<mask>") + len(self.sequences[i].replace("<mask>", ""))
                )
                for i in range(len(self.sequences))
            ]

    def filter_out_non_enzymes(self):
        original_count = len(self.ids)
        pass_indices = (self.labels_1[:, 0] == 0).nonzero(as_tuple=True)[0]
        self.embeddings = self.embeddings[pass_indices]
        self.labels_1 = self.labels_1[pass_indices]
        self.labels_2 = self.labels_2[pass_indices]
        self.labels_3 = self.labels_3[pass_indices]
        self.labels_4 = self.labels_4[pass_indices]
        self.ids = [self.ids[i] for i in pass_indices]
        self.sequences = [self.sequences[i] for i in pass_indices]
        print(f"Non-enzymes filtered: {original_count} -> {len(self.ids)}")

    def filter_out_missing_levels(self):
        original_count = len(self.ids)
        mask_counts = torch.zeros(self.labels_1.shape[0], dtype=torch.int64)
        mask_counts += (self.labels_1.sum(dim=1) < 0).long()
        mask_counts += (self.labels_2.sum(dim=1) < 0).long()
        mask_counts += (self.labels_3.sum(dim=1) < 0).long()
        mask_counts += (self.labels_4.sum(dim=1) < 0).long()
        mask_counts = mask_counts == 0
        pass_indices = mask_counts.nonzero(as_tuple=True)[0]
        self.embeddings = self.embeddings[pass_indices]
        self.labels_1 = self.labels_1[pass_indices]
        self.labels_2 = self.labels_2[pass_indices]
        self.labels_3 = self.labels_3[pass_indices]
        self.labels_4 = self.labels_4[pass_indices]
        self.ids = [self.ids[i] for i in pass_indices]
        self.sequences = [self.sequences[i] for i in pass_indices]
        print(f"Force all levels filtered: {original_count} -> {len(self.ids)}")

    def filter_out_non_clean_labels(self):
        id_ec_train, ec_id_dict_train = get_ec_id_dict(CLEAN_DATA_FOLDER / "split100.csv")
        clean_ecs = ec_id_dict_train.keys()
        level_idx_to_ec = pkl.load(open(SWISSPROT_EC_FOLDER / IDX_TO_EC_LEVEL, "rb"))
        ec_to_idx = {v[3:]: k for k, v in level_idx_to_ec["level_4"].items()}
        level_4_mask = [ec_to_idx[ec] for ec in clean_ecs if ec in ec_to_idx]
        original_count = len(self.ids)
        split_indices = self.label_split_indices(labels_4_mask=level_4_mask)
        self.embeddings = self.embeddings[split_indices]
        self.labels_1 = self.labels_1[split_indices]
        self.labels_2 = self.labels_2[split_indices]
        self.labels_3 = self.labels_3[split_indices]
        self.labels_4 = self.labels_4[split_indices]
        self.ids = [self.ids[i] for i in split_indices]
        self.sequences = [self.sequences[i] for i in split_indices]
        print(f"Clean labels filtered: {original_count} -> {len(self.ids)}")

    @classmethod
    def from_file(cls,
        dataset_folder: Path,
        split: str,
        model: ModelType = ModelType.ESM3,
        omit_non_enzymes: bool = False,
        force_all_levels: bool = False,
        clean_labels_only: bool = False,
        include_mask_level: bool = False,
        clean_labels: bool = False
    ):
        with open(ECEmbeddingDataset._split_pid_file(dataset_folder, split), "rb") as f:
            ids = pkl.load(f)
        with open(ECEmbeddingDataset._split_seq_file(dataset_folder, split), "rb") as f:
            sequences = pkl.load(f)
        if clean_labels:
            clean_labels_file = ECEmbeddingDataset._split_clean_label_file(dataset_folder, split)
            if not clean_labels_file.exists():
                raise ValueError(f"Clean labels file {clean_labels_file} does not exist.")
            level_4_file = clean_labels_file
        else:
            level_4_file = ECEmbeddingDataset._split_label_file(dataset_folder, split, 4)
        return cls(
            embeddings=torch.load(ECEmbeddingDataset._split_embedding_file(dataset_folder, split, model)).cpu(),
            labels=[
                torch.load(ECEmbeddingDataset._split_label_file(dataset_folder, split, 1)).cpu(),
                torch.load(ECEmbeddingDataset._split_label_file(dataset_folder, split, 2)).cpu(),
                torch.load(ECEmbeddingDataset._split_label_file(dataset_folder, split, 3)).cpu(),
                torch.load(level_4_file).cpu()
            ],
            ids=ids,
            sequences=sequences,
            omit_non_enzymes=omit_non_enzymes,
            force_all_levels=force_all_levels,
            clean_labels_only=clean_labels_only,
            include_mask_level=include_mask_level
        )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        if "include_mask_level" in self.__dict__ and self.include_mask_level:
            return (
                self.embeddings[idx],
                self.labels_1[idx],
                self.labels_2[idx],
                self.labels_3[idx],
                self.labels_4[idx],
                self.mask_level[idx],
            )
        else:
            return (
                self.embeddings[idx],
                self.labels_1[idx],
                self.labels_2[idx],
                self.labels_3[idx],
                self.labels_4[idx]
            )

    @staticmethod
    def _split_embedding_file(dataset_folder: Path, split: str, model: ModelType):
        return dataset_folder / f"{model}_embeddings_{split}.pt"

    @staticmethod
    def _split_label_file(dataset_folder: Path, split: str, level: int):
        return dataset_folder / f"labels_{level}_{split}.pt"

    @staticmethod
    def _split_pid_file(dataset_folder: Path, split: str):
        return dataset_folder / f"pids_{split}.pkl"

    @staticmethod
    def _split_seq_file(dataset_folder: Path, split: str):
        return dataset_folder / f"seq_{split}.pkl"

    @staticmethod
    def _split_clean_label_file(dataset_folder: Path, split: str):
        return dataset_folder / f"clean_labels_{split}.pkl"

    @staticmethod
    def _split_enzyme_esm1_emb_file(dataset_folder: Path, split: str):
        return dataset_folder / f"esm1b_embeddings_{split}.pt"

    @staticmethod
    def _split_enzyme_esm2_emb_file(dataset_folder: Path, split: str):
        return dataset_folder / f"esm2_embeddings_{split}.pt"

    def write_to_folder(self, dataset_folder: Path, split: str, model: ModelType):
        embedding_path = ECEmbeddingDataset._split_embedding_file(dataset_folder, split, model)
        torch.save(self.embeddings, embedding_path)
        print(f"Saved embeddings to {embedding_path}")
        labels = [None, self.labels_1, self.labels_2, self.labels_3, self.labels_4]
        for level in range(1, 4 + 1):
            label_path = ECEmbeddingDataset._split_label_file(dataset_folder, split, level)
            torch.save(labels[level], label_path)
            print(f"Saved labels for level {level} to {label_path}")
        id_path = ECEmbeddingDataset._split_pid_file(dataset_folder, split)
        with open(id_path, "wb") as f:
            pkl.dump(self.ids, f)
        print(f"Saved ids for split {split} to {id_path}")
        seq_path = ECEmbeddingDataset._split_seq_file(dataset_folder, split)
        with open(seq_path, "wb") as f:
            pkl.dump(self.sequences, f)
        print(f"Saved sequences for split {split} to {seq_path}")


    @staticmethod
    def create_dataset(
        dataset_folder: Path, source_datasets: List[Path], split_ids: Dict[str, Set[str]], level_ec_to_idx: Dict[str, Dict[str, int]],
        model_type: ModelType, filter_length: int = 1022, n_masking_replicates: int = 0, resume=False
    ):
        if not dataset_folder.exists():
            raise ValueError(f"Dataset folder {dataset_folder} does not exist.")

        for folder in source_datasets:
            if not folder.exists():
                raise ValueError(f"Source dataset {folder} does not exist.")

        split_sequences = {
            split: []
            for split in split_ids
        }
        split_embeddings = {
            split: []
            for split in split_ids
        }
        split_labels = {
            split: {
                f"label_{i}": [] for i in range(1, 4 + 1)
            }
            for split in split_ids
        }
        num_classes = {
            f"label_{i}": len(level_ec_to_idx[f"level_{i}"])
            for i in range(1, 4 + 1)
        }
        split_id_sets = copy.deepcopy(split_ids)
        split_ids = {
            split: []
            for split in split_ids
        }
        split_ids = cast(Dict[str, List[int]], split_ids)

        if resume:
            for split, expected_ids in split_id_sets.items():
                pid_file = ECEmbeddingDataset._split_pid_file(dataset_folder, split)
                if pid_file.exists():
                    with open(pid_file, "rb") as f:
                        loaded_ids = pkl.load(f)
                    if n_masking_replicates > 0:
                        expected_ids = [i for i in expected_ids for _ in range(1 + n_masking_replicates)]
                    assert set(loaded_ids) == set(expected_ids), f"Resume check failed: IDs for split {split} do not match."
            for split in split_id_sets:
                seq_file = ECEmbeddingDataset._split_seq_file(dataset_folder, split)
                with open(seq_file, "rb") as f:
                    split_sequences[split] = pkl.load(f)
                for level in range(1, 5):
                    label_file = ECEmbeddingDataset._split_label_file(dataset_folder, split, level)
                    with open(label_file, "rb") as f:
                        split_labels[split][f"label_{level}"] = torch.load(f)
                pid_file = ECEmbeddingDataset._split_pid_file(dataset_folder, split)
                with open(pid_file, "rb") as f:
                    split_ids[split] = pkl.load(f)
        else:
            print("Loading datasets and splitting them.")

            # invert the splits dict to make processing faster
            id_to_split = {}
            for split, ids in split_id_sets.items():
                for id in ids:
                    if id not in id_to_split:
                        id_to_split[id] = []
                    id_to_split[id].append(split)

            for source_dataset in source_datasets:
                for shard_file in tqdm(list(source_dataset.glob("*.json")), desc=f"Processing examples for {source_dataset.name}"):
                    with open(shard_file.with_suffix(".pkl"), "rb") as f:
                        examples = pkl.load(f)
                    for example in examples:
                        # make sure we don't exceed the esm1b max length
                        if len(example["sequence"]) > filter_length:
                            continue

                        # find which split this sequence belongs to
                        split = id_to_split.get(example["id"], None)
                        if split is None:
                            continue
                        if len(split) > 1:
                            raise ValueError(f"Example {example['id']} belongs to multiple splits: {split}")
                        split = split[0]

                        # add to the split
                        examples_to_add = [example]
                        if n_masking_replicates > 0:
                            for _ in range(n_masking_replicates):
                                masked_example = copy.deepcopy(example)
                                masked_example["sequence"] = uniformly_mask(masked_example["sequence"])
                                examples_to_add.append(masked_example)
                        for exp in examples_to_add:
                            split_sequences[split].append(exp["sequence"])
                            for l in split_labels[split]:
                                label = exp[l] if exp[l] is not None else -torch.ones(num_classes[l], dtype=torch.float32)
                                split_labels[split][l].append(label)
                            split_ids[split].append(exp["id"])

            print("Saving the ids, sequences, and labels for each split")
            for split in tqdm(split_ids):
                id_path = ECEmbeddingDataset._split_pid_file(dataset_folder, split)
                with open(id_path, "wb") as f:
                    pkl.dump(split_ids[split], f)

                seq_path = ECEmbeddingDataset._split_seq_file(dataset_folder, split)
                with seq_path.open("wb") as f:
                    pkl.dump(split_sequences[split], f)

                for level in range(1, 4 + 1):
                    label_path = ECEmbeddingDataset._split_label_file(dataset_folder, split, level)
                    with open(label_path, "wb") as f:
                        labels = torch.stack(split_labels[split][f"label_{level}"])
                        torch.save(labels, f)
                print(f"Saved {len(split_ids[split])} samples for split {split}")

        for split in split_ids:
            id_count = len(split_ids[split])
            seq_count = len(split_sequences[split])
            label_counts = [len(split_labels[split][f"label_{i}"]) for i in range(1, 5)]
            assert id_count == seq_count == label_counts[0] == label_counts[1] == label_counts[2] == label_counts[3], (
                f"Length mismatch in split '{split}': ids({id_count}), sequences({seq_count}), labels({label_counts})"
            )
        if model_type == ModelType.ESM1:
            embedding_func = get_esm1b_embeddings
            model = ESM1b_Embedding()
            model.cuda()
            model.eval()
        elif model_type == ModelType.ESM3:
            embedding_func = get_esm3_embeddings
            model = ESM3.from_pretrained("esm3-sm-open-v1").cuda()
            model.eval()
            model.to(dtype=torch.float32)
        else:
            raise NotImplementedError(f"Model {model_type} not implemented.")


        for split in split_embeddings:
            embedding_path = ECEmbeddingDataset._split_embedding_file(dataset_folder, split, model_type)
            if embedding_path.exists() and resume:
                print(f"Resuming embedding creation for split: {split}")
                continue
            with open(ECEmbeddingDataset._split_seq_file(dataset_folder, split), "rb") as f:
                seqs = pkl.load(f)
                embeddings = embedding_func(model, seqs)
            torch.save(embeddings, embedding_path)
            print(f"Saved {len(seqs)} embeddings for split {split}")
            print(embedding_path)
        print("Dataset creation complete.")

    def split(self, indices):
        if "include_mask_level" not in self.__dict__:
            self.include_mask_level = False
        return ECEmbeddingDataset(
            embeddings=self.embeddings[indices],
            labels=[
                self.labels_1[indices],
                self.labels_2[indices],
                self.labels_3[indices],
                self.labels_4[indices]
            ],
            ids=[self.ids[i] for i in indices],
            sequences=[self.sequences[i] for i in indices],
            include_mask_level=self.include_mask_level
        )

    # Only include samples that only have EC numbers for the indices in labels_mask
    def label_split_indices(self,
            labels_1_mask: Optional[List[int]] = None,
            labels_2_mask: Optional[List[int]] = None,
            labels_3_mask: Optional[List[int]] = None,
            labels_4_mask: Optional[List[int]] = None
        ):
        def filter_indices(labels_mask, labels, indices):
            indices = set(indices)
            if labels_mask is not None:
                assert len(set(labels_mask)) == len(labels_mask), "Make sure you input the valid indices of ECs instead of a binary mask"
                screen_indices = [i for i in range(len(labels[0])) if i not in labels_mask]
                mask_indices = []
                for i in indices:
                    label = labels[i]
                    if len(torch.nonzero(label[screen_indices])) == 0:
                        mask_indices.append(i)
                    else:
                        pass
                indices = indices & set(mask_indices)
            return list(indices)
        indices = list(range(len(self.embeddings)))
        # reverse order because we expect fewer misses earlier??
        indices = filter_indices(labels_4_mask, self.labels_4, indices)
        indices = filter_indices(labels_3_mask, self.labels_3, indices)
        indices = filter_indices(labels_2_mask, self.labels_2, indices)
        indices = filter_indices(labels_1_mask, self.labels_1, indices)
        return indices

    def label_masks(self):
        def get_mask(labels):
            defined_indices = [i for i in range(len(labels)) if torch.all(labels[i] >= 0)]
            return torch.nonzero(labels[defined_indices].sum(dim=0)).flatten().tolist()
        return [
            get_mask(self.labels_1),
            get_mask(self.labels_2),
            get_mask(self.labels_3),
            get_mask(self.labels_4)
        ]


    @classmethod
    def join(cls, datasets: List):
        datasets = cast(List[ECEmbeddingDataset], datasets)
        return ECEmbeddingDataset(
            embeddings=torch.cat([dataset.embeddings.cpu() for dataset in datasets]),
            labels=[
                torch.cat([dataset.labels_1.cpu() for dataset in datasets]),
                torch.cat([dataset.labels_2.cpu() for dataset in datasets]),
                torch.cat([dataset.labels_3.cpu() for dataset in datasets]),
                torch.cat([dataset.labels_4.cpu() for dataset in datasets])
            ],
            ids=[id for dataset in datasets for id in dataset.ids],
            sequences=[seq for dataset in datasets for seq in dataset.sequences]
        )

    @classmethod
    def add_clean_labels(cls, dataset_folder: Path, split: str, model: ModelType, level_ec_to_idx: Dict[str, Dict[int, str]]):
        clean_label_file = cls._split_clean_label_file(dataset_folder, split)
        if clean_label_file.exists():
            print(f"Clean labels already exist for {split}. Skipping.")
            return
        dataset = cls.from_file(dataset_folder=dataset_folder, split=split, model=model)
        idx_to_ec = level_ec_to_idx["level_4"] # this is actually an idx_to_ec mapping
        ec_to_idx = {v: k for k, v in idx_to_ec.items()}
        # load the models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        esm1b = ESM1b_Embedding()
        esm1b.to(device)
        clean = LayerNormNet(512, 128, device, torch.float32)
        checkpoint = torch.load(CLEAN_MLP, map_location=device)
        clean.load_state_dict(checkpoint)
        clean.eval()
        clean.to(device)
        # clean labels are only level 4 and not for non-enzymes
        # filter inputs to get labels for
        print("Filtering sequences for clean labels")
        enzymes = (dataset.labels_1[:, 0] == 0)
        filtered_seqs = [(i, s) for i, s in enumerate(dataset.sequences) if enzymes[i] and "<mask>" not in s]
        seqs = [s for _, s in filtered_seqs]
        pids = [dataset.ids[i] for i, _ in filtered_seqs]
        print(f"Filtered sequences: {len(seqs)}")
        # get esm1 embeddings
        # this is assuming this is an esm3 dataset
        esm1_emb_file = cls._split_enzyme_esm1_emb_file(dataset_folder, split)
        if not esm1_emb_file.exists():
            esm_emb = get_esm1b_embeddings(esm1b, seqs, verbose=True)
            torch.save(esm_emb, esm1_emb_file)
        else:
            esm_emb = torch.load(esm1_emb_file, map_location=device)

        clean_emb = clean(esm_emb.to(device))
        clean_preds = []
        batch_size = 1024
        for i in tqdm(range(0, len(clean_emb), batch_size), desc="Getting clean labels"):
            clean_preds_batch, pred_probs_batch = clean_labels_probs_from_emb(clean_emb[i:i+batch_size], verbose=False)
            clean_preds.extend(clean_preds_batch)
        ec_to_idx = {k[3:]: v for k, v in ec_to_idx.items()} # strip the "EC:" prefix
        # P is pid, C is enzyme class
        assert len(clean_preds) == len(seqs), f"Clean preds length {len(clean_preds)} does not match seqs length {len(seqs)}"
        clean_multihot_PC = torch.zeros(len(seqs), len(ec_to_idx), dtype=torch.float32)
        print("Forming multihot labels")
        for i, pred in enumerate(clean_preds):
            for ec in pred:
                if ec in ec_to_idx:
                    clean_multihot_PC[i][ec_to_idx[ec]] = 1.0
            if torch.sum(clean_multihot_PC[i]) == 0:
                print(f"Warning: no clean label found for sequence {i} with pid {pids[i]}")
                clean_multihot_PC[i][0] = 1.0 # set to non-enzyme
        print("Num clean preds that became non-enzyme:", torch.sum(clean_multihot_PC[:, 0] == 1.0))
        sample_pid_idx = {p: i for i, p in enumerate(pids)}
        # E is all the masked non-enzymes
        seq_sample_idx_E = torch.tensor([sample_pid_idx[p] for i, p in enumerate(dataset.ids) if enzymes[i]])
        # make a copy of the original labels
        # mask in the new clean labels
        # S is all sequences in the dataset
        clean_labels_SC = torch.clone(dataset.labels_4)
        seq_sample_idx_EC = seq_sample_idx_E.unsqueeze(1).expand(-1, clean_labels_SC.shape[1])
        clean_labels_SC[enzymes] = torch.gather(
            input=clean_multihot_PC,
            dim=0,
            index=seq_sample_idx_EC
        )
        # save these at the right file location
        torch.save(clean_labels_SC, clean_label_file)
        print(f"Saved clean labels to {clean_label_file}")
