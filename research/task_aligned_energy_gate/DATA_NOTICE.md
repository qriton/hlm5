# Banking77 data notice

This experiment uses BANKING77 from PolyAI's official
`task-specific-datasets` repository at commit
`57ec275d8078af65b7731c2a98be812d844a6d6b`.

The source dataset is licensed under Creative Commons Attribution 4.0
International. Source CSV files are not copied into this repository. Derived
manifests retain the utterance text needed to make the registered split and
overlap rules auditable. Cite Casanueva et al., *Efficient Intent Detection
with Dual Sentence Encoders* (ACL NLP4ConvAI 2020), when using the dataset.

The official test split is sealed by the runner until a registered development
gate passes. Seven test rows whose Unicode-normalized, case-folded,
whitespace-collapsed text appears in official training are excluded from the
primary test population before any encoder access. The full 3,080-row official
score is secondary and may be computed only after the same test authorization.
