# Quant OS v1.0.0 Release Checklist

## Required gates

- [x] Full Python regression suite passes.
- [x] Core/API coverage meets the release threshold.
- [x] Frontend strict TypeScript check and build pass.
- [x] Python AST and bytecode compilation pass.
- [x] Static release safety checks pass.
- [x] Paper-order end-to-end tests pass.
- [x] Deterministic replay economics pass.
- [x] Randomized execution accounting invariants pass.
- [x] Broker disconnect and reconciliation fail-closed tests pass.
- [x] Malformed market-data boundary tests pass.
- [x] Live-trading interlock tests pass.
- [x] Repository contains no private key material or `.env` file.
- [x] Generated caches/training logs are removed from the release archive.
- [x] ZIP integrity and SHA-256 are verified.

## Explicitly not claimed by v1.0.0

- No real-money broker adapter is certified or enabled.
- No strategy is guaranteed profitable.
- No model is guaranteed never to overfit; the Right-Fit engine reduces and measures that risk.
- The tested-version manifest is not a full transitive dependency lockfile.
