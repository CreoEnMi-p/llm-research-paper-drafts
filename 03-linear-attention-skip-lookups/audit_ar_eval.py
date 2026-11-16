"""Heldout autoregressive answer-token exposure audit on two-phase MQAR.

At answer slots, feed a model's own argmax answer into all later forward passes.
All keys, original fact values and query positions remain fixed ground truth.
No hidden labels are available to the prefix forward pass. This is a targeted
exposure-bias diagnostic, not natural free-form language generation.

The oracle (answer-position write suppression) is a task-aware upper-bound
intervention applied ONLY to inference with a frozen trained checkpoint.
"""
import numpy as np
import torch


def audit_autoregressive(model, heldout, device, a, rows=96):
    if a.task != "mqar":
        raise ValueError("Only original two-phase MQAR is supported")
    seq, mask = heldout
    rows = min(rows, seq.shape[0])
    gt_np = seq[:rows].copy()
    mk_np = mask[:rows]
    positions = np.flatnonzero(mk_np[0]).astype(int) + 1
    if not np.all(mk_np == mk_np[0:1]):
        raise AssertionError("Two-phase MQAR should have common answer slots")
    if not len(positions) or int(positions[-1]) >= gt_np.shape[1] - 1:
        # The original rc.py may score a last answer at the final un-input
        # position; a prefix model still handles it, but full TF m(x[:,:-1])
        # must output the logit from the preceding input position.
        if not len(positions) or int(positions[-1]) >= gt_np.shape[1]:
            raise AssertionError("Invalid answer slot")
    target = torch.as_tensor(gt_np, device=device, dtype=torch.long)
    was_training = model.training
    old_orq = a.orq
    model.eval()

    def one_condition(oracle):
        a.orq = 0.0 if oracle else -1.0
        with torch.inference_mode():
            # Batch teacher forcing: score the same exact rows and answers,
            # retaining all earlier ground-truth answer tokens.
            tf_logits = model(target[:, :-1])
            tf_preds = tf_logits[:, positions - 1, :].argmax(-1)
            truth = target[:, positions]
            tf_ok = (tf_preds == truth)

            # Left-to-right rollout: previous answers are the model's outputs.
            rollout = target.clone()
            per_query = []
            all_ok = []
            for pos in positions:
                logits = model(rollout[:, :int(pos)])
                prediction = logits[:, -1, :].argmax(-1)
                success = prediction == target[:, int(pos)]
                all_ok.append(success)
                per_query.append(round(float(success.float().mean()), 6))
                rollout[:, int(pos)] = prediction
            roll_ok = torch.stack(all_ok, dim=1)
            assert torch.equal(roll_ok[:, 0], tf_ok[:, 0]), "First query cannot be exposure-dependent"
            # Verify no factual tokens / key queries were modified by rollout.
            keep_mask = torch.ones(target.shape[1], dtype=torch.bool, device=device)
            keep_mask[positions] = False
            assert torch.equal(rollout[:, keep_mask], target[:, keep_mask])
            return {
                "n_rows": rows,
                "n_queries": len(positions),
                "tf_acc": round(float(tf_ok.float().mean()), 6),
                "ar_acc": round(float(roll_ok.float().mean()), 6),
                "ar_full_sequence_acc": round(float(roll_ok.all(1).float().mean()), 6),
                "ar_by_query": per_query,
                "n_wrong_tf": int((~tf_ok).sum()),
                "n_wrong_ar": int((~roll_ok).sum()),
            }

    try:
        base = one_condition(False)
        oracle = one_condition(True) if a.mix and all(c == "g" for c in a.mix) else None
    finally:
        a.orq = old_orq
        model.train(was_training)
    return {"no_intervention": base, "suppress_answer_writes": oracle}
