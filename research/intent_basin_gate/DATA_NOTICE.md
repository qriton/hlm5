# CLINC150 Data Notice

The two JSON manifests in `data/` contain reformatted excerpts from the CLINC
`oos-eval` dataset introduced by Stefan Larson, Anish Mahendran, Joseph J.
Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld,
Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars.

- Source: https://github.com/clinc/oos-eval
- Pinned commit: `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- Paper: https://aclanthology.org/D19-1131/
- Source license: Creative Commons Attribution 3.0 Unported
- License text: https://creativecommons.org/licenses/by/3.0/legalcode

Changes made here: the official rows were converted into explicit provenance
objects with stable indices and sorted intent IDs; training rows were separated
from development/test queries; and five evaluation rows whose exact text also
appeared in training were excluded and recorded before model access. No text was
otherwise edited.
