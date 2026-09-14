# Quant OS Frontend

Step 10 introduces the first production-oriented analytics terminal in `apps/web`.

## Current UI

- responsive dark terminal shell;
- portfolio metric cards and normalized equity curve;
- asset-correlation, parameter-stability, and strategy/regime heatmaps;
- P&L calendar;
- regime/risk sidecar;
- explicit demo-data banner and paper-mode safety state.

The Step 10 UI is implemented with strict TypeScript and browser-native APIs so it can be verified
without downloading a frontend dependency graph. Step 11 will connect the terminal to live API/event
streams and expand the execution, intelligence, market, tutorial, and broker workflows.

## Build

```bash
cd apps/web
tsc -p tsconfig.json
```

Open `apps/web/index.html` through a static web server after compiling `src/` to `dist/`.
