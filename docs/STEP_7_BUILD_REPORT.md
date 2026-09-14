# Step 7 Build Report — ML / DL / Foundation Model Laboratory

Overall target after this step: **73%**.

Implemented:

- leakage-aware lagged-return datasets and chronological train/validation/test splits;
- train-only feature standardization;
- regression, directional and calibration metrics;
- simple zero/mean baselines that advanced models must beat;
- Ridge, logistic-regression and random-forest adapters;
- optional XGBoost, LightGBM and CatBoost adapters;
- model-agnostic permutation importance;
- Platt probability calibration;
- weighted prediction ensembles;
- versioned model registry and local JSONL experiment tracking;
- leakage-aware walk-forward model tournament integration with Step 5;
- validation-based PyTorch early stopping;
- LSTM, GRU, TCN, Transformer, PatchTST-style and iTransformer-style models;
- lazy Chronos, TimesFM and financial Kronos foundation-model catalog;
- API capability discovery at `/ml/capabilities`.

Safety/research invariant: no ML result can authorize live execution. A model must still pass the
Right-Fit research gate, portfolio/risk policy and future live-execution gate.

## Verification

- Full project tests: 315 passed, 0 failed.
- ML/DL focused tests: 38 passed, 0 failed.
- ML/DL focused coverage: 95%.
- XGBoost/LightGBM/CatBoost local compatibility smoke tests passed.
- No foundation checkpoint is fetched automatically.

- Overall measured Python coverage: 95%.
- AST parsing and Python compilation: PASS.
- 100-character Python source-line gate: PASS.
- Core editable-install + API app-dir import smoke: PASS.
