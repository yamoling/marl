serve:
    python scripts/serve.py

studio port="5000":
    python scripts/serve_studio.py --port {{ port }}

build-studio:
    #! /bin/bash
    cd src/studio/frontend
    bun ci
    bun run build

optuna:
    optuna-dashboard tunings/perspective.journal
