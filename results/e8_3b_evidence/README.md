# E8 3B fresh full-path ZCA evidence

This directory preserves the complete byte-identical receipt chain from
Leonardo job `51800915` (A100-SXM-64GB, exit 0, 3m49s, 2026-08-11). The
registered implementation commit is
`3ee23ba8a1451f3231254fe7f7f63b4e3422468d`.

The formal verdict is `PASS_3B_FULL_PATH_ZCA`. All 64 preregistered
CounterFact cases were non-baseline-correct and therefore edit-eligible. Raw
target-row values were deployment-admitted for 29/64 cases. The frozen
head-only ZCA value direction was deployment-admitted for 64/64, a gain of 35,
with no raw-admitted case lost. All 64 candidate facts opened their exact gate,
selected their own memory slot, and strictly predicted the target through the
released HLM5 multi-slot adapter and untouched ordinary BF16 head.

The primary arm opened zero false gates across 264 registered off-support
queries: 128 paraphrases, 128 neighborhood prompts, and eight neutrals. All 264
gate-closed outputs were bit-identical to baseline. Removing every active slot
then restored exact-zero deltas and bit-identical baseline logits on all 337
registered prompts. A new Python process reproduced both direct arms, both
memories, all 328 gate rows, rollback, every scientific object, and every
full-logit hash exactly.

The result licenses only the bounded claim in `verdict.json`: a fixed
single-token, exact-key HLM5 adapter on the pinned frozen SmolLM3-3B-Base and
registered A100/BF16 runtime. It does not establish factual truth,
natural-language routing, paraphrase generalization, multi-token editing,
generation quality, PPL neutrality, cross-model transfer, trained-HLM
scaling, attention replacement, legal compliance, or commercial readiness.

SHA-256:

- `preflight.json`: `db200d971b28770b499bcf177ae46d29d6f90e005cab6c355e64481dc8cf5ce2`
- `execution_receipt.json`: `13e63826c9c83d5a7a9abb36e87880993f70f1450ca45b2695c34a2433704777`
- `attempt.json`: `4d2fc5381faef1c3b99316994eec90b7984c413c216403a02c06ee9581c10533`
- `admission.json`: `96d6dd8f8528d38de52d559df86f4228b508de4487cf9ea3efd50b63adccc7b0`
- `result.json`: `ec064b4d38c1abb9d88f99bd1c138b01cbfe4f9b15dbdf28edbe415477e678b6`
- `verdict.json`: `5e376c4c065ecf7e7602d3e81984e6425175876bc763ffd2a427684a6b9d7deb`
