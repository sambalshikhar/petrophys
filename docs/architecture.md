---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Architecture

Petrophys is organized by processing stage at the top level and by physical
problem inside `interpretation` and `evaluation`. Each feature has one home,
even when `import petrophys as pp` re-exports it.

## Package map

Generated from the installed package. Public names come from each package's
`__all__`; dependencies come from the imports that run when it loads.

```{code-cell} python
:tags: [remove-cell]
import pandas as pd

pd.set_option("display.max_colwidth", None)
```

```{code-cell} python
:tags: [hide-input]
import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import petrophys

roles = {
    "api": "The front door: pp.read, pp.interpret, pp.plot",
    "model": "Data structures: wells, curves, core, formations, units",
    "io": "Read and write LAS, DLIS, CSV, ASCII and Excel; geology and core tables",
    "preprocessing": "Standardization, normalization, quality control, imputation",
    "interpretation": "Physical properties from measurements",
    "evaluation": "Reservoir decisions: net pay, rock quality, flow units, summaries",
    "workflow": "Dependency-aware sequencing and method routing",
    "reporting": "Delivery packages, audit reports, LLM context",
    "visualization": "Log panels, crossplots, specialist displays",
    "datasets": "Bundled samples and repository fixtures",
    "mcp": "Model Context Protocol server",
    "exceptions": "Warnings and errors raised by Petrophys",
}
root = Path(petrophys.__file__).parent


def load_time_imports(name):
    """Other petrophys packages imported at module level by this package."""
    files = (root / name).rglob("*.py") if (root / name).is_dir() else [root / f"{name}.py"]
    found = set()
    for path in files:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("petrophys."):
                found.add(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                found.update(alias.name.split(".")[1] for alias in node.names if alias.name.startswith("petrophys."))
    return sorted(found - {name, "_internal", "version", "exceptions"})


rows = [{"Package": "petrophys (pp)", "What it holds": "The names most users need",
         "Public names": len(petrophys.__all__), "Sub-packages": "", "Imports when loaded": ""}]
for module in pkgutil.iter_modules(petrophys.__path__):
    if module.name.startswith("_") or module.name == "version":
        continue
    package = importlib.import_module(f"petrophys.{module.name}")
    public = getattr(package, "__all__", None)
    if public is None:  # a module without __all__: count the public functions and classes it defines
        public = [name for name, value in vars(package).items() if not name.startswith("_")
                  and (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ == package.__name__]
    children = [child.name for child in pkgutil.iter_modules(package.__path__)
                if child.ispkg and not child.name.startswith("_")] if module.ispkg else []
    summary = (package.__doc__ or "").strip().splitlines()
    rows.append({
        "Package": module.name,
        "What it holds": roles.get(module.name) or (summary[0] if summary else ""),
        "Public names": len(public),
        "Sub-packages": ", ".join(children),
        "Imports when loaded": ", ".join(load_time_imports(module.name)),
    })
pd.DataFrame(rows).style.hide(axis="index")
```

## Dependency direction

When a package loads, it imports only from the rows below it. So the science
never needs workflows or plotting, and the model needs no other package.

```text
api, mcp                         entry points: pp.read, pp.interpret, pp.plot, MCP
   ↓
workflow   visualization   reporting
   ↓
evaluation
   ↓
interpretation   preprocessing   io
   ↓
model                            (datasets holds bundled files and stands apart)
```

A few functions import upward when called, never at load time. For example,
`Well.from_las` uses `io`. Shared private helpers live in `petrophys._internal`
and are not public API.

## How a run moves through the packages

| Stage | Package | Result |
| --- | --- | --- |
| Read | `io` | A `Well` with source curves, header and attached geology |
| Prepare | `preprocessing` | Standard names and units, quality flags |
| Calculate | `interpretation` | Clay volume, porosity, saturation, permeability and more |
| Decide | `evaluation` | Rock quality, flow units, net pay, summaries |
| Sequence | `workflow` | The steps above, routed by the inputs present, with a trace |
| Deliver | `visualization`, `reporting` | Plots, LAS/JSON exports, reports and context |

## Contributing

Where new code belongs, and how to test and document it, is in
[CONTRIBUTING.md](https://github.com/sambalshikhar/petrophys/blob/main/CONTRIBUTING.md).

## Go deeper

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Orientation and setup
:link: tutorials/part_a/00_orientation_and_setup
:link-type: doc

A tour of the packages and the bundled data.
:::

:::{grid-item-card} The workflow, step by step
:link: tutorials/part_b/10_dynamic_workflow_glass_box
:link-type: doc

How `workflow` calls the science packages, step by step.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public function and class, by package.
:::
::::
