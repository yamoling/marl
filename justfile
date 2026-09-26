serve:
    python scripts/serve.py

studio port="5000":
    python scripts/serve_studio.py --port {{port}}

dashboard:
    optuna-dashboard tunings/perspective.journal
