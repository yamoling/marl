# MARL
## Requirements
- poetry
- python >=  3.10
- torch (does not work well with poetry, you should install it with pip as shown on the pytorch website)

## Getting started
```bash
$ poetry install
$ poetry shell
(your-venv) $ python src/main.py
```

## Web UI to inspect your experiments
After cloning the repo, you can serve the files either in development mode with hot-reloading or in production mode, which implies transpiling the sources explicitly. You need bun, node or deno to be installed to transpile. The below example assumes that you have [Bun](https://bun.sh/) installed.

Serve the files in production mode:
```bash
$ cd src/ui
$ bun install
$ bun run build # Build the sources to src/ui/dist.
$ cd ../..      # Go back to the root of the marl.
$ python src/serve.py
```

To serve the files in development mode, you need two terminals.
```bash
$ cd src/ui && bun run serve  # In one terminal
$ python src/serve.py         # In an other terminal
```
