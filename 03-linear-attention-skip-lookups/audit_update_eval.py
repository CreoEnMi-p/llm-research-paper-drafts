"""Fresh-heldout teacher-forced and greedy answer-only rollout diagnostics.

The stream includes external fact writes whose tokens are always supplied
truthfully at inference time; only answer tokens to *queries* are generated.
This tests continual-memory retrieval, not arbitrary generation.

For rewrite: Q2 'changed' keys have different latest values from their first
appearance, so predicting the original value is a measurable stale-memory error.
For alias: Q2 'changed' keys are newly introduced, with a value already
associated to an old key. Both require accepting a necessary new association.
"""
import numpy as np
import torch
import torch.nn.functional as F
from audit_update_task import layout


def update_eval(model, heldout, device, a):
    seq_np, mask_np = heldout
    B = len(seq_np)
    N = a.pairs
    W, Q1, Q2, bd = layout(a.L, N)
    x = torch.tensor(seq_np, dtype=torch.long, device=device)
    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
    qa_pos = torch.arange(bd[1]+1, bd[2], 2, device=device)
    qb_pos = torch.arange(bd[3]+1, bd[4], 2, device=device)
    all_pos = list(qa_pos.cpu().tolist() + qb_pos.cpu().tolist())
    first_keys = x[:, :2*N:2]
    first_vals = x[:, 1:2*N:2]
    secondary_keys = x[:, bd[2]:bd[3]:2]
    secondary_vals = x[:, bd[2]+1:bd[3]:2]
    final_keys = x[:, bd[3]:bd[4]:2]
    changed = (final_keys[:, :, None] == secondary_keys[:, None, :]).any(-1)
    assert changed.sum().item() == B*(Q2//2)
    assert not changed.logical_not().all()

    # Validate labels against the latest visible facts; this is a task
    # generator correctness check, not an input available to the model.
    fmatch = (x[:, qa_pos-1, None] == first_keys[:, None, :])
    assert fmatch.sum(-1).eq(1).all()
    first_answers = (fmatch.long() * first_vals[:, None, :]).sum(-1)
    assert torch.equal(first_answers, x[:, qa_pos])

    initial_match = (final_keys[:, :, None] == first_keys[:, None, :])
    new_match = (final_keys[:, :, None] == secondary_keys[:, None, :])
    old_ans = (initial_match.long() * first_vals[:, None, :]).sum(-1)
    new_ans = (new_match.long() * secondary_vals[:, None, :]).sum(-1)
    later_ans = torch.where(changed, new_ans, old_ans)
    assert torch.equal(later_ans, x[:, qb_pos])
    if a.task == "mqar_rewrite":
        assert initial_match.sum(-1).eq(1).all()
        assert torch.all(old_ans[changed] != later_ans[changed])
    else:
        assert not initial_match[changed].any()
        for v in secondary_vals.cpu().numpy().ravel():
            assert v >= a.tv//2

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        tf_logits = model(x[:, :-1])
        gt_ans = x[:, torch.tensor(all_pos, device=device)]
        tf_query_logits = tf_logits[:, torch.tensor(all_pos,device=device)-1, :]
        tf_argmax = tf_query_logits.argmax(-1)

        # The model reads each query KEY and writes its own predicted VALUE.
        # Later external facts and query keys stay grounded.
        generated = x.clone()
        ar_pred = []
        for pos in all_pos:
            logits = model(generated[:, :pos])
            pred = logits[:, -1, :].argmax(-1)
            ar_pred.append(pred)
            generated[:, pos] = pred
        ar_argmax = torch.stack(ar_pred, 1)
        assert torch.equal(tf_argmax[:, 0], ar_argmax[:, 0])
        keep = torch.ones(x.shape[1],dtype=torch.bool,device=device)
        keep[torch.tensor(all_pos,device=device)] = False
        assert torch.equal(x[:, keep], generated[:, keep])

        groups = {
            "query_A": torch.cat([
                torch.ones((B,Q1),dtype=torch.bool,device=device),
                torch.zeros((B,Q2),dtype=torch.bool,device=device)
            ], 1),
            "query_B_changed": torch.cat([
                torch.zeros((B,Q1),dtype=torch.bool,device=device),
                changed
            ], 1),
            "query_B_unchanged": torch.cat([
                torch.zeros((B,Q1),dtype=torch.bool,device=device),
                ~changed
            ], 1),
            "all_queries": torch.ones((B,Q1+Q2),dtype=torch.bool,device=device),
        }

        def by_group(pred):
            result = {}
            for name, g in groups.items():
                ok = pred[g].eq(gt_ans[g])
                result[name] = {
                    "n": int(g.sum()),
                    "accuracy": round(float(ok.float().mean()), 6),
                }
            return result

        tf_g, ar_g = by_group(tf_argmax), by_group(ar_argmax)
        result = {
            "task": a.task,
            "heldout_seed": a.update_heldout_seed,
            "n_rows": B,
            "pairs": N, "updates": W, "queries_A": Q1, "queries_B": Q2,
            "teacher_forced": tf_g,
            "autoregressive": ar_g,
            "teacher_forced_nll": round(float(F.cross_entropy(tf_query_logits.reshape(-1,a.tv),
                                                        gt_ans.reshape(-1))), 6),
            "autoregressive_all_answers_correct": round(
                float((ar_argmax==gt_ans).all(-1).float().mean()), 6),
        }
        if a.task == "mqar_rewrite":
            stale_ar = (ar_argmax[:, Q1:] == old_ans) & changed
            stale_tf = (tf_argmax[:, Q1:] == old_ans) & changed
            result["old_value_prediction_rate_changed"] = {
                "teacher_forced": round(float(stale_tf.sum()/changed.sum()), 6),
                "autoregressive": round(float(stale_ar.sum()/changed.sum()), 6)
            }
        else:
            # The second facts use existing values, but new distinct keys.
            result["alias_value_in_old_set_fraction"] = round(
                float((secondary_vals[...,None] == first_vals[:,None,:]).any(-1).float().mean()), 6
            )
    model.train(was_training)
    return result
