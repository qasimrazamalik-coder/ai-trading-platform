# Security and Trading Safety

Quant OS v1.0.0 is a research and paper-trading release. Real-money adapters are not enabled
in the shipped adapter catalog.

## Safety invariants

- Live trading defaults to disabled.
- Enabling the live-trading setting requires production mode and the explicit acknowledgement
  `I_UNDERSTAND_REAL_MONEY_RISK`.
- The independent risk engine sits between strategy intent and execution.
- Broker reconciliation fails closed when connectivity or remote state cannot be verified.
- Financial boundary values use `Decimal`; binary floats are rejected for orders and canonical
  market-data price/size fields.
- Paper orders use idempotent client order IDs.
- No API credentials, broker tokens, or private keys are stored in the repository.

## Production deployment

Do not expose the local control-plane API directly to the public internet. Put authentication,
TLS, authorization, rate limiting, secret management, durable event storage, and network policy
in front of any future broker-enabled deployment.

A real-money adapter must remain disabled until its documented API, reconciliation behavior,
failure recovery, permissions, and broker-specific order semantics have dedicated integration
and sandbox tests.
