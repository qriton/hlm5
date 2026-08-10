# E7b 3B native-admission evidence

This directory preserves the complete byte-identical receipt chain from
Leonardo job `51795959` (A100-SXM-64GB, exit 0, 2026-08-10).

The formal verdict is `FAIL_3B_NATIVE_ADMISSION` because exact affine geometry
reached 893 of 1,200 targets, below the preregistered minimum of 960. The
ordinary BF16 head admitted all 893 reachable targets, and the new-process
replay reproduced every decision, recorded value, and native-logit hash.

SHA-256:

- `preflight.json`: `9a55b4d5f162f6f8b8c47a8ac11156544d952ee0d55921fb1408272c42522c66`
- `execution_receipt.json`: `afe9609e2af892bfbb98f5248864b2978d4fa46aaf969cddd57271240e60ac09`
- `attempt.json`: `c3f445dbb20b0223ee065104231d73b31250a1c3a40f3f6059c4d079a76b790b`
- `admission.json`: `60249139741d79c3995205bc09c04e0f68067155098abff7d4c1d9d08e828b39`
- `result.json`: `9a5232b33f8828195794cb137ee2ec83e10ec8af524418abe3608cf0ac2d4d11`
- `verdict.json`: `5f5e35f71d432d2df0287c7dec1350e8b53c3dae7bea0f025634536a3a6fd016`
