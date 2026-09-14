# Execution

Production code lives in `packages/core/src/quant_os_core/execution`.

The Step 11 execution boundary includes an idempotent OMS, deterministic paper broker, position and
order reconciliation, adapter capability contracts and Decimal-based accounting. Live venue
connectors remain disabled until users explicitly configure documented broker/exchange APIs.
