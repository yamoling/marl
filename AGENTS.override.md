# AGENTS.md overrides

## Machine-specific instructions

On this machine, you are running with 8 old GPUs (1080Ti). To accomodate this, pytorch must be installed with version 2.7 at most. Any version superior to that will break at run-time. When a python project is managed with a `pyproject.toml` and `uv`, make sure to use the `legacy-gpu` extra (or define one if it does not exist) such that the proposer versino of pytorch is used. For instance:

```toml
[project]
# ...
dependencies = ["torch"]

[project.optional-dependencies]
legacy-gpu = ["torch<2.8"]
modern-gpu = ["torch"]

# ...
[tool.uv]
conflicts = [[{ extra = "legacy-gpu" }, { extra = "modern-gpu" }]]
```
