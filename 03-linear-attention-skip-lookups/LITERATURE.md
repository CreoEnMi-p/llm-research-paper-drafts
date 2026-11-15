# Literature check: predicted-skip write (2026-11-15)

Status: PARTLY PUBLISHED (principle known, exact gate new). Own web search (2 extended searches at gate 3, one more pass on 2026-11-15 evening) plus the gate-0 haiku paper search (2026-11-15 19:10). The haiku agent listed Gated DeltaNet (2412.06464), Titans (2501.00663), DeltaProduct (2502.10297), Gated DeltaNet-2 (2605.22791) and Longhorn (2407.14207) as closest, plus Erase-then-Delta Attention (2606.26560); I read those abstracts. None gates the write on the previous token's read matching the current value.

Question searched: does any published method scale a linear-attention or fast-weight write by how well the read made at the previous token already predicts the value being written at this token?

## Three closest papers

| paper | what it does | how predicted-skip differs |
|---|---|---|
| Gated DeltaNet-2 (arXiv 2605.22791) | learned channel-wise write and erase gates computed from the current token only | our gate depends on the memory's own read (S_c q_{t-1}) compared with v_t; a token-only gate cannot know whether the memory already holds the value. Used as competitor e in every grid cell. |
| HOLA, "A Hippocampus for Linear Attention" (arXiv 2607.02303) | uses the delta residual (surprise) to choose which tokens go into an exact softmax side cache | surprise picks cache entries; the recurrent write itself is not gated. Our gate lowers the write into the state. |
| AURA (arXiv 2606.02775) | fast-weight memory for robot policies with a learned surprise gate trained on action error | gate is driven by a task error signal; ours is driven by a read-value cosine inside the layer, with no extra loss. |

Also checked and different: Gated DeltaNet (2412.06464, learned scalar decay), DeltaProduct (2502.10297, several delta steps per token, no write gate), Longhorn (2407.14207, step size from an online recall objective at the current key), Erase-then-Delta Attention (2606.26560, erase along a learned direction before the delta write), Titans (surprise = gradient of the memory loss at the current key), the delta rule (writes v - S k at the current key; our gate uses the previous token's query), Muir (arXiv 2611.08376, sensitivity-allocated low-rank gates plus a matched-filter readout), Dynamic Compression (arXiv 2608.17896, re-scanning), Gated Slot Attention.

Sources: https://arxiv.org/pdf/2605.22791, https://arxiv.org/html/2607.02303v1, https://arxiv.org/pdf/2606.02775, https://arxiv.org/html/2611.08376, https://arxiv.org/pdf/2608.17896

## Prior-work check (manager, 2026-11-15 evening): partly published

The exact gate `beta *= exp(-gamma relu(cos(S_c q_{t-1}, v_t)))` is unpublished. The principle "write less when the memory already predicts the value" is published. Abstracts read on arXiv:

| paper | what it does | relation |
|---|---|---|
| PH-attention, "Attention as Conditioning" (arXiv 2508.08289) | Pearce-Hall associability state: the write strength is scaled by a running average of the prediction error `|v - S k|`, read through the current key | closest; same principle, current-key error, built into the update rule |
| Lattice (arXiv 2504.05646) | each memory slot is updated only with the part orthogonal to its current state | same principle, slot-level |
| LoLA (arXiv 2505.23666) | self-recall error decides which pairs go to a sparse exact cache | same principle, cache policy |
| Kaiser et al., "Learning to Remember Rare Events" (arXiv 1703.03129) | skips a new memory entry when the nearest neighbour already returns the right value | same principle, key-value memory module |
| "How Linear Attention Remembers" (arXiv 2609.33093) | in pretrained GLA and GDN, interference from later writes is shaped by their overlap with existing memory | supports the diagnosis |

Reframed claims (PAPER.md section 2): (1) the diagnosis that re-written answers cause most recall interference; (2) a cheap gate that reuses the chunked kernel's read; (3) it composes with GDN-2 and the delta rule. The paper no longer claims that current-key methods cannot see a just-returned value: the `k_t` read ties the `q_{t-1}` read in our model (0.159 against 0.164, within noise).

Earlier verdict from own search (superseded): no paper gates the write on the previous token's read matching the current value. This still holds for the exact gate.
