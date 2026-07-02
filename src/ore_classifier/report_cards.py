from __future__ import annotations

from collections.abc import Mapping, Sequence


def render_model_card(
    *,
    model_name: str,
    intended_use: str,
    provenance: Mapping[str, object],
    metrics: Mapping[str, object] | None = None,
    limitations: Sequence[str] | None = None,
) -> str:
    lines = [
        f"# Model Card: {model_name}",
        "",
        "## Intended Use",
        intended_use,
        "",
        "## Provenance",
        *_mapping_lines(provenance),
        "",
        "## Metrics",
        *(_mapping_lines(metrics or {"status": "not reported"})),
        "",
        "## Limitations",
        *_list_lines(limitations or ["Not yet reviewed by a domain expert."]),
        "",
    ]
    return "\n".join(lines)


def render_dataset_card(
    *,
    dataset_name: str,
    composition: Mapping[str, object],
    labels: Mapping[str, object],
    recommended_use: str,
    caveats: Sequence[str] | None = None,
) -> str:
    lines = [
        f"# Dataset Card: {dataset_name}",
        "",
        "## Composition",
        *_mapping_lines(composition),
        "",
        "## Labels",
        *_mapping_lines(labels),
        "",
        "## Recommended Use",
        recommended_use,
        "",
        "## Caveats",
        *_list_lines(caveats or ["Pseudo-labels are not expert geological ground truth."]),
        "",
    ]
    return "\n".join(lines)


def render_run_fact_sheet(
    *,
    run_id: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
    parameters: Mapping[str, object],
    checks: Mapping[str, object] | None = None,
) -> str:
    lines = [
        f"# Run Fact Sheet: {run_id}",
        "",
        "## Inputs",
        *_mapping_lines(inputs),
        "",
        "## Parameters",
        *_mapping_lines(parameters),
        "",
        "## Outputs",
        *_mapping_lines(outputs),
        "",
        "## Checks",
        *_mapping_lines(checks or {"status": "not reported"}),
        "",
    ]
    return "\n".join(lines)


def _mapping_lines(values: Mapping[str, object]) -> list[str]:
    return [f"- `{key}`: {value}" for key, value in values.items()]


def _list_lines(values: Sequence[str]) -> list[str]:
    return [f"- {value}" for value in values]
