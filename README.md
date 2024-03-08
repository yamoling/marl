# MARL
This repository contains a variety of Multi-Agent Reinforcement Learning (MARL) algorithms. Its purpose is to develop new algorithms and it is not (yet) intended to be a library to include from an other project.

`marl` is strongly typed and has high code quality standards. Any contribution to this repository is expected to exhibit a similar quality.
`marl` comes with a web interface to visualise the results of your experiments (more info down below).

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
$ bun intall
$ bun run build # Build the sources to src/ui/dist.
$ cd ../..      # Go back to the root of the marl.
$ python src/serve.py
```

To serve the files in development mode, you need two terminals.
```bash
$ cd src/ui && bun run serve  # In one terminal
$ python src/serve.py         # In an other terminal
```
