# ML / DL Research Lab

Quant OS treats machine learning as a research component, not an automatic source of alpha.
Every advanced model must be compared against simple baselines and evaluated on chronological,
out-of-sample data.

## Leakage controls

- Features are built from information available at prediction time only.
- Targets begin strictly after the feature observation time.
- Standardization is fitted on training data only.
- Walk-forward tournaments use the Step 5 time-series split engine and support a gap.
- Validation data, not training loss, controls neural-network early stopping.
- Model checkpoints are never downloaded implicitly.

## Implemented families

Classical adapters include Ridge, logistic regression, random forests, XGBoost, LightGBM and
CatBoost. Third-party boosters remain optional dependencies.

PyTorch research architectures include LSTM, GRU, temporal convolution, Transformer,
PatchTST-style and iTransformer-style forecasters.

Foundation-model integration is deliberately lazy. Chronos, TimesFM and financial Kronos are
represented by catalog/adaptor metadata so licensing, checkpoint provenance and local resource
requirements can be reviewed before any model is loaded.

## Governance

The model registry records monotonically increasing versions, model family, artifact URI and an
optional training-data hash. Experiments can be written to JSONL for local, auditable research.
A later persistence layer will move these records into the platform database without changing the
core contracts.
