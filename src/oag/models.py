import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.sdk.api import LogitsConfig
from esm.utils.sampling import _BatchedESMProteinTensor
from CLEAN.esm import pretrained
from CLEAN.model import LayerNormNet
from CLEAN.esm.architectures.esm1 import ProteinBertModel
from oag.constants import CHECKPOINT_FOLDER
from typing import cast, List

class ESM1b_Embedding(nn.Module):

    ESM_VERSION = "esm1b_t33_650M_UR50S"
    ESM_MAX_LEN = 1022 # since first token is <cls> and even though it's not supposed to, it seems they are including <eos> at some point...

    def __init__(self, include_eos: bool = False):
        super().__init__()
        self.esm1b, self.esm1b_alphabet = pretrained.load_model_and_alphabet(ESM1b_Embedding.ESM_VERSION)
        self.esm1b = cast(ProteinBertModel, self.esm1b)
        self.esm1b_alphabet.append_eos = include_eos # we don't want to add <eos> to the sequences
        self.has_eos = include_eos
        esm1b_batch_converter = self.esm1b_alphabet.get_batch_converter()
        self.esm1b_tokenizer = lambda seq_batch: esm1b_batch_converter([(s, s) for s in seq_batch])[2] # it outputs labels, strs, tokens and expects an input and label string as input

    def forward(self, s):
        if isinstance(s, str):
            s = [s]
        # s is a batch of sequences
        # it returns the embeddings for each sequence in the batch
        self.esm1b.eval()
        with torch.no_grad():
            device = next(self.esm1b.parameters()).device
            esm1b_tokens = self.esm1b_tokenizer(s).to(device)
            esm1b_embeddings = self.esm1b(esm1b_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        return esm1b_embeddings, (esm1b_tokens != 1) # return mask when not tokenized so the user can isolate the right tokens

    def tokenize(self, s):
        return self.esm1b_tokenizer(s)

    def forward_pretokenized(self, s):
        self.esm1b.eval()
        with torch.no_grad():
            device = next(self.esm1b.parameters()).device
            s = s.to(device)
            esm1b_embeddings = self.esm1b(s, repr_layers=[33], return_contacts=False)["representations"][33]
        return esm1b_embeddings

class ESM3Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ESM3.from_pretrained("esm3-sm-open-v1").cuda()
        self.model.eval()
        self.model.to(dtype=torch.float32)

    def forward(self,seqs: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        logits_config = LogitsConfig(sequence=True, return_embeddings=True)    
        tokenized_seqs = [self.model.tokenizers.sequence.encode(seq) for seq in seqs]
        max_len = max(len(seq) for seq in tokenized_seqs)
        padded_seqs_BR = [seq + [1] * (max_len - len(seq)) for seq in tokenized_seqs]  # 1 is the padding token for ESM3
        padded_seqs_BR = torch.tensor(padded_seqs_BR).to(device)
        padded_seqs_BR = _BatchedESMProteinTensor(sequence=padded_seqs_BR)
        logits = self.model.logits(padded_seqs_BR, logits_config)

        embeddings_LRE = []
        for tokenized_seq, embedding_RE in zip(tokenized_seqs, logits.embeddings):
            embedding_RE = embedding_RE[:len(tokenized_seq)]
            embeddings_LRE.append(embedding_RE.mean(dim=0))
        return torch.stack(embeddings_LRE)

    def forward_tokenized(self, tokenized_seqs: torch.Tensor) -> torch.Tensor:
        logits_config = LogitsConfig(sequence=True, return_embeddings=True)    
        device = tokenized_seqs.device
        max_len = max(len(seq) for seq in tokenized_seqs)
        padded_seqs_LR = [torch.concat([seq, torch.ones((max_len - len(seq),), device=seq.device, dtype=torch.long)]) for seq in tokenized_seqs]  # 1 is the padding token for ESM3
        padded_seqs_BR = torch.stack(padded_seqs_LR)
        padded_seqs_BR = _BatchedESMProteinTensor(sequence=padded_seqs_BR)
        logits = self.model.logits(padded_seqs_BR, logits_config)

        embeddings_LRE = []
        for tokenized_seq, embedding_RE in zip(tokenized_seqs, logits.embeddings):
            embedding_RE = embedding_RE[:len(tokenized_seq)]
            embeddings_LRE.append(embedding_RE.mean(dim=0))
        return torch.stack(embeddings_LRE)

class EmbeddingClassifier(nn.Module):
    ESM3_EMBEDDING_SIZE = 1536
    CLEAN_EMBEDDING_SIZE = 128
    def __init__(self, levels_dict, label_mask=None, time_disc=0.1):
        super().__init__()
        level_4_len = len(levels_dict["level_4"])
        # if time_disc <= 0 or time_disc > 1:
        if time_disc <= 0:
            raise ValueError("time_disc must be in (0, 1]") # since need to include the endpoints for 0 and 1 in time
        if time_disc > 1:
            print("Warning, only one model will be used, time_disc effectively set to 1.0")
            n_models = 1
        else:
            n_models = int(1 / self.time_disc) + 1
        self.time_disc = time_disc
        self.output_dim = len(label_mask) if label_mask is not None else level_4_len
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Linear(EmbeddingClassifier.ESM3_EMBEDDING_SIZE, 2 * EmbeddingClassifier.ESM3_EMBEDDING_SIZE),
                nn.GELU(),
                nn.Linear(2 * EmbeddingClassifier.ESM3_EMBEDDING_SIZE, 2 * EmbeddingClassifier.ESM3_EMBEDDING_SIZE),
                nn.GELU(),
                nn.Linear(2 * EmbeddingClassifier.ESM3_EMBEDDING_SIZE, self.output_dim),
            ) for _ in range(n_models) 
        ])
        
        self.embed_time = nn.Embedding(len(self.model), len(self.model)) # to convert time to the model to use
        self.embed_time.weight = nn.Parameter(torch.eye(len(self.model)), requires_grad=False)

        if label_mask is not None:
            output_to_ec = torch.zeros((level_4_len, self.output_dim))
            for i_out, i_ec in enumerate(label_mask):
                output_to_ec[i_ec, i_out] = 1.0
            output_to_ec = nn.Parameter(output_to_ec, requires_grad=False)
            non_label_mask = nn.Parameter(torch.ones((level_4_len)), requires_grad=False)
            non_label_mask[label_mask] = 0.0
            non_label_mask *= -1e6

            self.output_to_ec = nn.Linear(len(label_mask), level_4_len, bias=True)
            self.output_to_ec.weight = output_to_ec
            self.output_to_ec.bias = non_label_mask
        else:
            self.output_to_ec = None

        self.aggregate = ECAggregate(levels_dict)

    def forward(self, x, mask_frac): # E: ensemble, B: batch, C: class
        if len(self.model) == 1:
            model_preds_BC = self.model[0](x)
        else:
            t = 1 - mask_frac
            t_bins = (t / self.time_disc).int()
            # pred_weighting_BE = self.embed_time(t_bins)
            # model_preds_EBC = torch.stack([m(x) for m in self.model])
            # model_preds_BEC = torch.transpose(model_preds_EBC, 0, 1)
            # model_preds_BC = torch.einsum("bec,be->bc", model_preds_BEC, pred_weighting_BE)
            model_preds_BC = torch.zeros((x.shape[0], self.output_dim), device=x.device)
            for i in range(len(self.model)):
                mask_bin = t_bins == i
                if mask_bin.sum() > 0:
                    model_preds_BC[mask_bin] = self.model[i](x[mask_bin])
        logits_BC = nn.LogSoftmax(dim=1)(model_preds_BC)
        return logits_BC

    def forward_hierarchical(self, x, mask_frac):
        t = 1 - mask_frac
        class_logits = self(x, t)
        if self.output_to_ec is None:
            ec_logits = class_logits
        else:
            ec_logits = self.output_to_ec(class_logits)
        four_level_logits = self.aggregate(ec_logits)
        return four_level_logits

def create_hierarchy_matrix(lower_level: int, higher_level: int, levels_dict: dict) -> torch.Tensor:
    lower_key = f"level_{lower_level}"
    higher_key = f"level_{higher_level}"
    lower_levels = levels_dict[lower_key]
    higher_levels = levels_dict[higher_key]
    matrix = torch.zeros((len(lower_levels), len(higher_levels)))
    for higher_idx, higher_label in higher_levels.items():
        prefix = higher_label.split("-")[0]
        for lower_idx, lower_label in lower_levels.items():
            if lower_label.startswith(prefix):
                matrix[lower_idx, higher_idx] = 1.0
    return matrix

class ECAggregate(nn.Module):
    def __init__(self, levels_dict):
        super().__init__()
        self.levels_dict = levels_dict
        l4_to_l3 = create_hierarchy_matrix(4, 3, self.levels_dict)
        l3_to_l2 = create_hierarchy_matrix(3, 2, self.levels_dict)
        l2_to_l1 = create_hierarchy_matrix(2, 1, self.levels_dict)
        self.register_buffer("l4_to_l3", l4_to_l3) # makes sure the tensors are moved to the right device
        self.register_buffer("l3_to_l2", l3_to_l2)
        self.register_buffer("l2_to_l1", l2_to_l1)
        self.l4_to_l3.requires_grad_(False)
        self.l3_to_l2.requires_grad_(False)
        self.l2_to_l1.requires_grad_(False)
    
    def forward(self, x, **kwargs):
        l4 = torch.exp(x)
        l3 = l4 @ self.l4_to_l3
        l2 = l3 @ self.l3_to_l2
        l1 = l2 @ self.l2_to_l1
        outputs = [torch.log(l1 + 1e-10), torch.log(l2 + 1e-10), torch.log(l3 + 1e-10), torch.log(l4 + 1e-10)]
        return outputs

class Classifier(nn.Module):
    def __init__(self, classifier_hash, levels_dict, label_mask, time_disc):
        super().__init__()
        self.embed = ESM3Embedding()
        self.classifier = EmbeddingClassifier(levels_dict, label_mask, time_disc)
        checkpoint = get_best_checkpoint(classifier_hash)
        print(f"Loading classifier from {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location="cpu")
        self.classifier.model.load_state_dict(checkpoint)

    def forward(self, seqs, t=None): # batch, sequence, alphabet
        esm_embed_BH = self.embed(seqs)
        mask_frac = (1 - t) if t is not None else None
        x_LBE = self.classifier.forward_hierarchical(esm_embed_BH, mask_frac=mask_frac) # level, batch, ecn
        return x_LBE

    def forward_tokenized(self, tokenized_seqs, t=None): # batch, sequence, alphabet
        esm_embed_BH = self.embed.forward_tokenized(tokenized_seqs)
        mask_frac = (1 - t) if t is not None else None
        x_LBE = self.classifier.forward_hierarchical(esm_embed_BH, mask_frac=mask_frac)
        return x_LBE

    def forward_one_hot(self, x_BSA, t=None): # batch, sequence, alphabet
        if len(x_BSA) > 1:
            # because haven't added taking out tokenized elements below
            raise NotImplementedError("Batch size > 1 not implemented yet")
        # have to use a modified forward pass that is differentiable through the one-hot encoding
        esm_embed_BSH = self.embed.model.forward_one_hot(sequence_one_hot=x_BSA).embeddings
        esm_embed_BH = esm_embed_BSH.mean(dim=1) # mean over the sequence length
        mask_frac = (1 - t) if t is not None else None
        x_BSA = self.classifier.forward_hierarchical(esm_embed_BH, mask_frac=mask_frac)
        return x_BSA

    def forward_embedded(self, esm_embed_BH, t=None): # batch, sequence, alphabet
        mask_frac = (1 - t) if t is not None else None
        x_BSA = self.classifier.forward_hierarchical(esm_embed_BH, mask_frac=mask_frac)
        return x_BSA

def get_best_checkpoint(experiment_hash):
    checkpoint_folder = CHECKPOINT_FOLDER / experiment_hash
    if not checkpoint_folder.exists():
        raise ValueError(f"Checkpoint folder {checkpoint_folder} does not exist.")
    checkpoint_files = list(checkpoint_folder.glob("*.pt"))
    def checkpoint_better_than(a, b):
        a_f1 = float(a.stem.split("_")[-1])
        b_f1 = float(b.stem.split("_")[-1])
        a_epoch = int(a.stem.split("_")[-3])
        b_epoch = int(b.stem.split("_")[-3])
        if a_f1 == b_f1:
            return a_epoch > b_epoch
        else:
            return a_f1 > b_f1
    checkpoint = checkpoint_files[0]
    for c in checkpoint_files:
        if checkpoint_better_than(c, checkpoint):
            checkpoint = c
    return checkpoint