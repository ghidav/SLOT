"""
SLOT optimization for vLLM v1 (0.17.x).

Called from a patched model_runner.py's sample_tokens() method.
Modifies hidden_states in execute_model_state before sampling.
"""

import os

import torch


def _compute_loss_and_grad(hidden_states, input_ids, lm_head_weight, params):
    """Manual forward + backward. Returns (loss, gradient)."""
    h = hidden_states + params
    logits = h @ lm_head_weight.T

    shift_logits = logits[:-1, :].contiguous()
    shift_labels = input_ids[1:].contiguous()

    # Softmax
    max_l = torch.max(shift_logits, dim=-1, keepdim=True)[0]
    exp_l = torch.exp(shift_logits - max_l)
    probs = exp_l / torch.sum(exp_l, dim=-1, keepdim=True)
    del max_l, exp_l

    flat_labels = shift_labels
    valid_mask = (flat_labels >= 0).float()
    valid_labels = torch.clamp(flat_labels, min=0)

    # Loss: -mean(log(p[y]))
    idx = torch.arange(flat_labels.size(0), device=flat_labels.device)
    masked_p = probs[idx, valid_labels] * valid_mask
    n_valid = torch.sum(valid_mask)
    loss = -torch.sum(torch.log(masked_p + 1e-10)) / (n_valid + 1e-10)
    del idx, masked_p

    # Gradient: d(CE)/d(logits) = (p - one_hot) / n_valid
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, valid_labels.unsqueeze(1),
                     valid_mask.to(dtype=one_hot.dtype).unsqueeze(1))
    d_logits = (probs - one_hot) / (n_valid + 1e-10)
    del one_hot, probs

    # Backprop through lm_head: d_hidden = d_logits @ W
    d_hidden = torch.matmul(d_logits, lm_head_weight)
    grad = d_hidden.mean(dim=0)
    del d_logits, d_hidden

    return loss, grad


def _optimize_adamw(hidden_states, input_ids, lm_head_weight, steps, lr):
    """AdamW SLOT optimizer."""
    hidden_dim = hidden_states.shape[-1]
    params = torch.zeros(hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
    adam_m = torch.zeros_like(params)
    adam_v = torch.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.95, 1e-6
    weight_decay = 1e-5

    for step in range(1, steps + 1):
        loss, grad = _compute_loss_and_grad(hidden_states, input_ids, lm_head_weight, params)

        adam_m = beta1 * adam_m + (1 - beta1) * grad
        adam_v = beta2 * adam_v + (1 - beta2) * (grad**2)

        m_hat = adam_m / (1 - beta1**step)
        v_hat = adam_v / (1 - beta2**step)

        update = lr * m_hat / (torch.sqrt(v_hat) + eps)
        params = params - update - lr * weight_decay * params

    return params


def _optimize_lbfgs(hidden_states, input_ids, lm_head_weight, steps, lr):
    """L-BFGS SLOT optimizer with two-loop recursion + Armijo line search."""
    hidden_dim = hidden_states.shape[-1]
    params = torch.zeros(hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    m = min(steps, 10)
    s_hist, y_hist, rho_hist = [], [], []
    c1 = 1e-4
    max_ls = 10

    def compute(p):
        return _compute_loss_and_grad(hidden_states, input_ids, lm_head_weight, p)

    prev_grad = None
    prev_s = None
    loss, grad = compute(params)

    for k in range(steps):
        # 1. Update history from completed step
        if prev_grad is not None and prev_s is not None:
            y_k = (grad - prev_grad).float()
            ys = torch.dot(y_k, prev_s)
            if ys > 1e-10:
                if len(s_hist) >= m:
                    s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)
                s_hist.append(prev_s)
                y_hist.append(y_k)
                rho_hist.append(1.0 / ys)

        # 2. Two-loop recursion
        q = grad.clone().float()
        n_hist = len(s_hist)
        alphas = [None] * n_hist

        for i in range(n_hist - 1, -1, -1):
            alphas[i] = rho_hist[i] * torch.dot(s_hist[i], q)
            q = q - alphas[i] * y_hist[i]

        if n_hist > 0:
            gamma = torch.dot(y_hist[-1], s_hist[-1]) / (torch.dot(y_hist[-1], y_hist[-1]) + 1e-10)
            r = gamma * q
        else:
            r = q

        for i in range(n_hist):
            beta = rho_hist[i] * torch.dot(y_hist[i], r)
            r = r + s_hist[i] * (alphas[i] - beta)

        direction = -r.to(params.dtype)

        # 3. Backtracking line search (Armijo)
        step_size = lr
        dir_dot_grad = torch.dot(direction.float(), grad.float())
        if dir_dot_grad >= 0:
            direction = -grad
            dir_dot_grad = -torch.dot(grad.float(), grad.float())

        accepted_grad = None
        for _ in range(max_ls):
            new_params = params + step_size * direction
            new_loss, new_grad = compute(new_params)
            if new_loss <= loss + c1 * step_size * dir_dot_grad:
                accepted_grad = new_grad
                break
            step_size *= 0.5
        else:
            new_params = params + step_size * direction
            new_loss, accepted_grad = compute(new_params)

        prev_s = (new_params - params).float()
        prev_grad = grad.float()
        params = new_params
        loss = new_loss
        grad = accepted_grad

    return params


def _get_lm_head_weight(model_runner):
    """Get (and cache) full lm_head weight."""
    if hasattr(model_runner, '_slot_lm_head_weight'):
        return model_runner._slot_lm_head_weight

    tp_size = int(os.environ.get("tensor_parallel_size_my", "1"))
    local_weight = model_runner.model.lm_head.weight.detach()

    if tp_size == 1:
        model_runner._slot_lm_head_weight = local_weight
    else:
        import torch.distributed
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        first_rank = torch.distributed.get_rank() - rank
        group = torch.distributed.new_group(list(range(first_rank, first_rank + tp_size)))
        shards = [torch.empty_like(local_weight) for _ in range(tp_size)]
        torch.distributed.all_gather(shards, local_weight, group=group)
        model_runner._slot_lm_head_weight = torch.cat(shards, dim=0)

    return model_runner._slot_lm_head_weight


def slot_optimize_hidden_states(model_runner):
    """
    Called from patched sample_tokens(). Optimizes SLOT delta on prefill
    requests and returns the modified execute_model_state tuple.
    """
    chot_steps = int(os.environ.get("CHOT_STEPS", "0"))
    chot_lr = float(os.environ.get("CHOT_LR", "0.1"))
    chot_optimizer = os.environ.get("CHOT_OPTIMIZER", "adamw")
    tp_size = int(os.environ.get("tensor_parallel_size_my", "1"))
    print(f"[SLOT] called: steps={chot_steps}, lr={chot_lr}, opt={chot_optimizer}", flush=True)

    (
        input_batch,
        model_inputs,
        attn_metadata,
        slot_mappings_by_layer,
        hidden_states,
        aux_hidden_states,
        kv_connector_output,
        num_tokens_across_dp,
    ) = model_runner.execute_model_state

    if not isinstance(hidden_states, torch.Tensor):
        return model_runner.execute_model_state

    lm_head_weight = _get_lm_head_weight(model_runner)
    input_ids = input_batch.input_ids[:input_batch.num_tokens]
    num_scheduled = input_batch.num_scheduled_tokens
    req_ids = input_batch.req_ids

    optimize_fn = _optimize_lbfgs if chot_optimizer == "lbfgs" else _optimize_adamw

    if not hasattr(model_runner, '_slot_deltas'):
        model_runner._slot_deltas = {}

    # Clean up deltas for finished requests
    active_req_ids = set(req_ids)
    for rid in list(model_runner._slot_deltas.keys()):
        if rid not in active_req_ids:
            del model_runner._slot_deltas[rid]

    offset = 0
    for i in range(len(num_scheduled)):
        n_tokens = int(num_scheduled[i])
        rid = req_ids[i]

        if n_tokens <= 1:
            # Decode — apply cached delta
            if rid in model_runner._slot_deltas:
                hidden_states[offset:offset + n_tokens] += model_runner._slot_deltas[rid]
            offset += n_tokens
            continue

        # Prefill — optimize delta
        req_hidden = hidden_states[offset:offset + n_tokens]
        req_input_ids = input_ids[offset:offset + n_tokens]

        delta = optimize_fn(req_hidden, req_input_ids, lm_head_weight, chot_steps, chot_lr)
        scaled_delta = delta / tp_size

        hidden_states[offset:offset + n_tokens] += scaled_delta
        model_runner._slot_deltas[rid] = scaled_delta

        offset += n_tokens

    return (
        input_batch,
        model_inputs,
        attn_metadata,
        slot_mappings_by_layer,
        hidden_states,
        aux_hidden_states,
        kv_connector_output,
        num_tokens_across_dp,
    )
