import random
import torch
import torch.nn.functional as F
import numpy as np
import cvxpy as cp
from typing import Callable, Optional
from oag.logging import Logger
from enum import Enum
from oag.utils import print_categorical_distribution

class STEP_CONFIG(Enum):
    ALL = "all"
    FIXED = "fixed"
    CUSTOM = "custom"


def masked_sampling_step(
    batched_esm_forward: Callable[[torch.Tensor], None], # ESM LOGITS THINGY OR SMTHG TODO
    xt: torch.Tensor,
    # t: torch.Tensor,
    device: torch.device,
    mask_tok: int,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor], torch.Tensor] # embedding to logits
    ] = None,
    guide_temp: float = 1.0,
    use_tag: bool = False,
    x1_temp: float = 1.0,
    eps: float = 1e-20,
    stochasticity: float = 0.0,
    save_logits: bool = False,
    data_guide: bool = False,
    logger: Optional[Logger] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Generates samples using "fake gillespie, which is just autoregressive"

    This function implements the core sampling algorithm for discrete flow matching with masking noise.
    It supports both predictor guidance and predictor-free guidance, and includes options for
    purity-based sampling and padding token handling.

    Args:
        denoising_model: Model that takes (x_t: [B,D], t: [B]) and returns logits [B,D,S]
        batch_size: Number of samples to generate in parallel
        S: Size of categorical state space (vocabulary size)
        D: Dimension of each sample (sequence length)
        device: Device to run generation on
        dt: Time step size for Euler integration
        mask_idx: Token index used for masking. Defaults to S-1
        pad_idx: Optional token index used for padding
        predictor_log_prob: Optional predictor function for guided sampling that takes (x,t)
            and returns log p(y|x,t) of shape [B]
        cond_denoising_model: Optional conditional model for predictor-free guidance that
            takes (x,t) and returns logits [B,D,S]
        guide_temp: Temperature for guidance (1 / gamma). Lower = stronger guidance
        stochasticity: Amount of stochastic noise in sampling
        use_tag: Whether to use Taylor approximation for predictor guidance
        argmax_final: Whether to use argmax for any remaining masked tokens at end
        max_t: Maximum time value to run sampling
        x1_temp: Temperature for softmax of model logits
        do_purity_sampling: Whether to weight sampling by prediction confidence
        purity_temp: Temperature for purity-based sampling weights
        num_unpadded_freq_dict: Optional dict mapping num unpadded tokens to frequencies
        eps: Small constant for numerical stability

    Returns:
        numpy.ndarray: Generated samples of shape [batch_size, D]
    """
    if logger is None:
        logger = Logger(fields=[])

    if not ((xt.ndim == 2 and xt.shape[0] == 1) or xt.ndim == 1):
        raise ValueError(
            f"xt should be of shape (1, D) or (D,) but got {xt.shape}"
        )
    if stochasticity >= 1.0 or stochasticity < 0.0:
        raise ValueError(
            f"Stochasticity should be in [0, 1.0) but got {stochasticity}."
        )

    xt_D = xt.squeeze()
    xt_D = xt_D.to(device)

    # uniformly sample the next position to unmask
    unmask_idx = np.random.choice(
        np.where(xt_D.cpu().numpy() == mask_tok)[0], 1, replace=False
    )[0]

    # We need the logits from the current sequence and embeddings from all possible
    # mutations to get the classifier predictions at the sampled position.
    # Therefore, we will batch them all together to save inference time.
    tok_alphabet_A = torch.arange(start=4, end=24, step=1, device=device, dtype=xt.dtype)
    n_mut = len(tok_alphabet_A)
    xt_mutated_BD = torch.tile(
        xt_D[None, ...], (1 + n_mut, 1)
    )
    xt_mutated_BD[1:, unmask_idx] = tok_alphabet_A[None, :]

    logits_BDA, embeds_BDH = batched_esm_forward(xt_mutated_BD, save_logits=False) # using logger instead
    logits_BDA, embeds_BDH = logits_BDA.to(device), embeds_BDH.to(device)

    logits_A = logits_BDA[0, unmask_idx, tok_alphabet_A] # need to get rid of the non-AA tokens
    # unconditional term
    p_xtilde_g_x_A = F.softmax(logits_A / x1_temp, dim=-1) # here I use g for given
    logger.log(Logger.Field.P_XTILDE_G_X_A, p_xtilde_g_x_A, 0)

    embeds_ADH = embeds_BDH[1:]
    embeds_AH = embeds_ADH.mean(dim=1)
    # HACK: made a mistake and parameterized the model with t = 1 - mask_frac
    # instead of uniformly sampling times, so now I need to use this
    mask_frac = (torch.sum(xt_D == mask_tok) - 2) / (xt_D.shape[0] - 2) # 2 for cls, eos
    t_model = (1 - mask_frac) * torch.ones((embeds_AH.shape[0],), device=device)
    if not data_guide:
        predictor_logits_A = predictor_log_prob(embeds_AH, t=t_model, save_logits=False, all_logits=False) # A is the batch dimension for this function call; don't want to save logits here because they are the lookahead logits
    else:
        predictor_logits_A, predictor_logits_AC = predictor_log_prob(embeds_AH, t=t_model, save_logits=False, all_logits=True) # A is the batch dimension for this function call; don't want to save logits here because they are the lookahead logits
    # "reward" term
    q_y_g_xtilde_A = torch.exp(predictor_logits_A)
    logger.log(Logger.Field.Q_Y_G_XTILDE_A, q_y_g_xtilde_A, 0)

    # "expected" reward over current policy term (bayes rule denominator)
    p1_y_g_x = torch.dot(q_y_g_xtilde_A, p_xtilde_g_x_A)
    logger.log(Logger.Field.P1_Y_G_X, p1_y_g_x, 0)
    # if you want to anneal the guidance term, uncomment this
    # log_R = torch.log(p_y_g_xtilde_A) - torch.log(p_y_g_x)
    # log_R *= 10
    # log_R *= 1 / guide_temp
    # log_p_xtilde_g_x_y_A = torch.log(p_xtilde_g_x_A) + log_R
    # p_xtilde_g_x_y_A = F.softmax(log_p_xtilde_g_x_y_A, dim=-1)
    p_guide_xtilde_g_x_y_A = q_y_g_xtilde_A * p_xtilde_g_x_A / p1_y_g_x
    logger.log(Logger.Field.PGUIDE_XTILDE_G_X_A, p_guide_xtilde_g_x_y_A, 0)
    # R = p_y_g_xtilde_A / p_y_g_x
    # p_xtilde_g_x_y_A = R * p_xtilde_g_x_A
    

    # Denoiser Divergence
    if not data_guide:
        logit = predictor_log_prob(embeds_BDH[:1].mean(dim=1), t=t_model[:1], save_logits=save_logits, all_logits=False)
        q_y_g_x = torch.exp(logit)
        logger.log(Logger.Field.Q_Y_G_X, q_y_g_x, 0)
        p_transition_xtilde_g_x_y = p_guide_xtilde_g_x_y_A
    if data_guide:
        logit, logits_C = predictor_log_prob(embeds_BDH[:1].mean(dim=1), t=t_model[:1], save_logits=save_logits, all_logits=True)
        q_y_g_x = torch.exp(logit)
        q_y_g_x_C = torch.exp(logits_C.squeeze())
        q_y_g_xtilde_CA = torch.exp(predictor_logits_AC).T
        logger.log(Logger.Field.Q_Y_G_X, q_y_g_x, 0)
        logger.log(Logger.Field.Q_Y_G_X_C, q_y_g_x_C, 0)
        logger.log(Logger.Field.Q_Y_G_XTILDE_CA, q_y_g_xtilde_CA, 0)
        # p_guide_y_g_x_C = q_y_g_xtilde_CA @ p_xtilde_g_x_A
        # delta_C = q_y_g_x_C - p_guide_y_g_x_C
        # Q_CA = q_y_g_xtilde_CA
        # b_C = delta_C + (q_y_g_xtilde_CA @ p_guide_y_g_x_C)
        N = q_y_g_xtilde_CA.shape[1]
        Q = q_y_g_xtilde_CA.cpu().numpy()
        b = q_y_g_x_C.cpu().numpy()
        q = cp.Variable(N)
        # uniform = np.ones(N) / N  # Uniform distribution for regularization
        # lambda_reg = 0.0001
        # objective = cp.Minimize(
        #     cp.sum_squares(Q @ q - b)
        #     + lambda_reg * cp.sum_squares(q - uniform)
        # )
        objective = cp.Maximize(cp.sum(cp.entr(q)))
        constraints = [
            cp.sum_squares(Q @ q - b) <= 1e-6,
            q >= 0,
            cp.sum(q) == 1
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        singular_values_A = torch.svd(q_y_g_xtilde_CA, compute_uv=False).S
        condition_number = singular_values_A[0] / singular_values_A[-1]
        logger.log(Logger.Field.KAPPA_Q, condition_number, 0)
        logger.log(Logger.Field.ENTROPY_Q, prob.value, 0)
        logger.log(Logger.Field.LSQ_RESIDUAL, np.sum(np.abs(Q @ q.value - b)), 0)

        q_xtilde_g_x_A = torch.tensor(q.value).cuda()
        q_xtilde_g_x_A = torch.clamp(q_xtilde_g_x_A, min=0.0, max=1.0)
        q_xtilde_g_x_A = q_xtilde_g_x_A / torch.sum(q_xtilde_g_x_A)  # Renormalize after clipping
        logger.log(Logger.Field.Q_XTILDE_G_X_A, q_xtilde_g_x_A, 0)
        eps_q = (q_xtilde_g_x_A + eps) / torch.sum(q_xtilde_g_x_A + eps)
        eps_p = (p_xtilde_g_x_A + eps) / torch.sum(p_xtilde_g_x_A + eps)
        D = torch.nn.functional.kl_div(
            torch.log(eps_q),
            torch.log(eps_p),
            log_target=True,
        )
        print(f"Generator divergence (D_KL(q, p)): {D}")
        print("p:", p_xtilde_g_x_A)
        print("q:", q_xtilde_g_x_A)
        print("q-p:", q_xtilde_g_x_A - p_xtilde_g_x_A)
        print("root-residual:", prob.value ** 0.5)
        print("condition number:", condition_number)
        q_xtilde_g_x_y = (q_y_g_xtilde_A * q_xtilde_g_x_A) / torch.sum(q_y_g_xtilde_A * q_xtilde_g_x_A)
        p_transition_xtilde_g_x_y = q_xtilde_g_x_y

    # if torch.abs(D) / p_y_g_x > 0.15:
    #     # Try sampling a different position to unmask
    #     # and with probability stochasticity, remask some position
    #     # to try to help. Note that for stochasticity > 0.5 there is
    #     # no guarantee of forward progress
    r = random.random()
    if r < stochasticity:
        if torch.sum(xt_D[1:-1] != mask_tok) > 0:
            mask_idx = 1 + np.random.choice(
                np.where(xt_D[1:-1].cpu().numpy() != mask_tok)[0], 1, replace=False
            )[0]
            xt_D[mask_idx] = mask_tok
        return xt_D.detach().cpu()

    # # HACK: going to threshold probabilities for mutations that are too bad
    # # adjust the threshold based on the error in q(y|x_t) - p(y|x_t)
    # # when the classifier isn't that trustworthy be okay with less optimal choices
    # # The idea is that p(y) at the next step shouldn't drop much below the last step if possible
    # threshold = min(max(0, (1 - 2 * torch.abs(D))) * q_y_g_x, max(q_y_g_xtilde_A))
    # p_xtilde_g_x_A[q_y_g_xtilde_A < threshold] = 0.0
    # p_xtilde_g_x_A /= torch.sum(p_xtilde_g_x_A)
    # # HACK: going to threshold probabilities for mutations that are too bad

    alphabet_offset = tok_alphabet_A[0].item()
    mutation = torch.multinomial(p_transition_xtilde_g_x_y, 1).squeeze()
    new_state = mutation + alphabet_offset
    assert new_state in tok_alphabet_A, f"new_state {new_state} not in tok_alphabet_A {tok_alphabet_A}"
    xt_D[unmask_idx] = new_state
    if data_guide:
        print(f"Original target probability: {q_y_g_x.item():.4f}")
    print(f"Mutated target probability: {q_y_g_xtilde_A[mutation]:.4f}, {(100 * q_y_g_xtilde_A[mutation] / max(q_y_g_xtilde_A)):.4f} of max available")

    # log updated p(x) and q(x) according to the sampled transition
    sample = logger.get_sample(0)
    if Logger.Field.P_X in sample:
        p_x = sample[Logger.Field.P_X][-1]
        p_xtilde = p_xtilde_g_x_A[mutation].item() * p_x
        logger.log(Logger.Field.P_X, p_xtilde, 0)
    if Logger.Field.Q_X in sample:
        q_x = sample[Logger.Field.Q_X][-1]
        q_xtilde = q_xtilde_g_x_A[mutation].item() * q_x
        logger.log(Logger.Field.Q_X, q_xtilde, 0)

    return xt_D.detach().cpu()