import torch
import numpy as np
from oag.constants import HOME_FOLDER, DATA_FOLDER, DB_FOLDER, OUTPUT_FOLDER

def mask(seq, mask_num=None):
    if mask_num is None:
        mask_fraction = np.random.uniform(1, len(seq)) # not 0 because clean sequence gets added anyways
        mask_num = int(mask_fraction)
    masked_positions = np.random.choice(len(seq), mask_num, replace=False)
    residues = list(seq)
    for i in masked_positions:
        residues[i] = "<mask>"
    return ''.join(residues)

def decode(x_BS, model, padding=None): # batch and sequence length
    if padding is not None:
        raise NotImplementedError("Decoding with padded batch not implemented")
    protein_sequence = model.tokenizers.sequence.decode(x_BS[1:-1].detach().clone().int()) # shave off register tokens
    protein_sequence = protein_sequence.replace(" ", "") # Remove spaces from esm3 tokenizer output
    return protein_sequence

def print_categorical_distribution(probs, labels=None, width=50, symbol='█', zero_threshold=1e-6):
    """
    Print a horizontal bar chart of a categorical distribution to the terminal.
    
    Args:
        probs: Array-like of probabilities/values for each category
        labels: Optional list of labels for each category
        width: Maximum width of the bars in characters
        symbol: Character to use for drawing bars
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    probs = np.array(probs)
    max_prob = np.max(probs)
    
    if labels is None:
        labels = [f"Cat {i}" for i in range(len(probs))]

    # Filter for nonzero elements if they're less than 10% of distribution
    nonzero_mask = probs > zero_threshold
    nonzero_count = np.sum(nonzero_mask)
    if nonzero_count < 0.1 * len(probs):
        probs = probs[nonzero_mask]
        labels = [labels[i] for i in range(len(labels)) if nonzero_mask[i]]

    max_label_width = max(len(str(label)) for label in labels)
    
    for i, (prob, label) in enumerate(zip(probs, labels)):
        bar_length = int((prob / max_prob) * width)
        bar = symbol * bar_length
        print(f"{str(label):>{max_label_width}} | {bar} {prob:.3f}")