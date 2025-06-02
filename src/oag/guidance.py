import torch
import torch.nn.functional as F
from tqdm import tqdm
from oag.sampling import masked_sampling_step, STEP_CONFIG
from oag.logging import Logger
from oag.utils import decode
from typing import Tuple, List
from enum import Enum

def esm3_classifier_guidance(
    model,
    classifier,
    target_class: Tuple[int, int], # (level of class, class index)
    masked_sequences: List[str],
    x1_temp=0.7,
    stochasticity=0.0,
    guide_temp=1.0,
    use_tag=True,
    batch_size=4,
    save_paths=True,
    save_logits=True,
    num_classes=None,
    steps_strategy="esm",
    # num_steps=None,
    contrast=False,
    verbose=False,
    unconditional=False,
    data_guide=False,
    logger=None,
    **kwargs,
):
    """
    Main entry function for flow-matching sampling of ESM for inverse folding
    Wrapper function for sampling ESM inverse folding with rate matrix

    Returns:
        proteins (List[ESMProtein])
        tokens (List[torch.tensor])
    """
    if steps_strategy != STEP_CONFIG.ALL:
        raise NotImplementedError("Only ALL sampling strategy is implemented for now")

    if logger is None:
        logger = Logger(fields=[])

    tokenized_seqs = [model.tokenizers.sequence(seq)["input_ids"] for seq in masked_sequences]
    # TODO: make this batched using the padding frequency dictionary
    sampled_sequences = []
    if save_paths:
        sampled_paths = []
    if save_logits:
        logits_px = []
        logits_pyx = []

    # Define wrapper denoising function that works for a batch
    # denoising_model_func_raw = lambda xt, t: model.forward(sequence_tokens=xt).sequence_logits[..., :model.tokenizers.sequence.vocab_size] # to get rid of the extra logits at index 34-63
    denoising_model_func_raw = lambda xt, t: model.forward(sequence_tokens=xt)

    def denoising_model_func(xt_BD, t, save_logits, return_all=False):
        xt_BD[:, 0] = 0  # Set first position to <cls>
        xt_BD[:, -1] = 2  # Set last position to <eos>
        outputs = denoising_model_func_raw(xt_BD, t)
        logits_BSA = outputs.sequence_logits
        logits_BSA[:, 0, 0] = 0.0  # Set <cls> to 0
        logits_BSA[:, 0, 1:] = -float("inf")  # Mask out rest
        logits_BSA[:, -1, 2] = 0.0  # Set <eos> to 0
        logits_BSA[:, -1, :2] = -float("inf")  # Mask out rest
        logits_BSA[:, -1, 3:] = -float("inf")  # Mask out rest
        logits_BSA[:, 1:-1, 0:4] = -float("inf")  # Mask out special tokens
        logits_BSA[:, 1:-1, 24:] = -float("inf")  # Mask out <mask> and non-standard amino acids; for reference <mask> is 32 and <pad> is 1
        if return_all:
            return outputs
        return logits_BSA

    def batched_denoising_model_func(x_PD, save_logits): # proteins, dimension (sequence length)
        logits = []
        embeds = []
        for i in range(0, x_PD.shape[0], batch_size):
            x_BD = x_PD[i:i+batch_size]
            with torch.no_grad():
                outputs = denoising_model_func(x_BD, 0, save_logits, return_all=True)
            logits.append(outputs.sequence_logits.detach().clone().cpu())
            embeds.append(outputs.embeddings.detach().clone().cpu())
        logits_PDA = torch.cat(logits, dim=0)
        embeds_PDH = torch.cat(embeds, dim=0)
        return logits_PDA, embeds_PDH
    
    def predictor_log_prob(xt_BSA3, t, save_logits, all_logits): # batch, sequence, logits=dim-64 (not alphabet=dim-33)
        if unconditional: # hacky way to do unconditional sampling
            zeros = torch.zeros((num_classes,), requires_grad=True).to(xt_BSA3.device)
            if xt_BSA3.ndim == 3:
                preds_BC = torch.einsum("bsa,c->bc", xt_BSA3, zeros)
            elif xt_BSA3.ndim == 2:
                xt_BE = xt_BSA3 # this is coming from gillespie and is actually the batch of protein embeddings
                preds_BC = torch.einsum("be,c->bc", xt_BE, zeros)
            if save_logits:
                if xt_BSA3.ndim == 3: # don't have real saving for this yet, just store all zeros
                    logits_pyx[-1].append(preds_BC[0].detach().clone().cpu().squeeze())
                elif xt_BSA3.ndim == 2:
                    preds_LBC = classifier.forward_embedded(xt_BE, t=t) # actual logits for the sampling
                    preds_C_actual = preds_LBC[target_class[0]][0].detach().clone().cpu().squeeze()
                    logits_pyx[-1].append(preds_C_actual)
                    if verbose: print(f"Current Class Prob: {torch.exp(preds_C_actual[target_class[1]])}")
            preds_B = preds_BC[:, target_class[1]]
            if all_logits:
                return preds_B, preds_C_actual
            else:
                return preds_B
        elif steps_strategy == STEP_CONFIG.ALL:
            xt_BH = xt_BSA3 # input is actually the batch of protein embeddings
            with torch.no_grad():
                preds_LBC = classifier.forward_embedded(xt_BH, t=t)
            for i in range(len(preds_LBC)):
                preds_LBC[i] = F.log_softmax(preds_LBC[i] / guide_temp, dim=-1)
        # elif use_tag:
        #     preds_LBC = classifier.forward_one_hot(xt_BSA3, t=t) 
        # else:
        #     xt_BS = xt_BSA3 # if not using tag, you actually just get the tokenized sequence
        #     with torch.no_grad():
        #         preds_LBC = classifier.forward_tokenized(xt_BS.long(), t=t) # and we can conver to long since no need for gradients

        preds_B = preds_LBC[target_class[0]][:, target_class[1]] # second zero because only one sequence per batch
        if verbose: print(f"Current Class Prob: {torch.exp(preds_B)}")

        if contrast:
            contrast_pred_B = preds_LBC[target_class[0]][:, 0] # zero is the non-enzyme class
            preds_B = preds_B - contrast_pred_B

        if all_logits:
            return preds_B, preds_LBC[target_class[0]]
        else:
            return preds_B

    # Call fm_utils sampling
    for i, seq in enumerate(tokenized_seqs):
        mask_idx = model.tokenizers.sequence.mask_token_id
        xt = torch.Tensor(seq).long()
        pbar = tqdm(total=len(seq) - 2, desc=f"Gillespie sampling")
        # break_ct = 0
        logger.log(Logger.Field.P_X, 1.0, i)
        logger.log(Logger.Field.Q_X, 1.0, i)
        while not torch.all(xt[1:-1] != mask_idx):
            protein_sequence = decode(xt, model)
            logger.log(Logger.Field.NOISED_SEQUENCES, protein_sequence, i)
            logger.batch_start(i)
            xt = masked_sampling_step(
                xt=xt,
                batched_esm_forward=batched_denoising_model_func,
                device=model.device,
                mask_tok=mask_idx,
                x1_temp=x1_temp,
                predictor_log_prob=predictor_log_prob,
                guide_temp=guide_temp,
                use_tag=use_tag,
                stochasticity=stochasticity,
                save_logits=save_logits,
                data_guide=data_guide,
                logger=logger,
            )
            logger.batch_end()
            xt = torch.tensor(xt)
            progress = (xt[1:-1] != mask_idx).sum()
            pbar.update(progress.item() - pbar.n)
            # break_ct += 1
            # if break_ct == 10:
            #     break
        sampled_seq = xt.unsqueeze(0) # add batch dimension like the other sampling method requires
        protein_sequence = decode(sampled_seq.squeeze(0).long(), model)
        logger.log(Logger.Field.SAMPLE_SEQUENCE, protein_sequence, i)
        sampled_sequences.append(protein_sequence)
    return sampled_sequences