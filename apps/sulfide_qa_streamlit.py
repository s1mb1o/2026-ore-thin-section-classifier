from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review binary sulfide / ore pipeline outputs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/inference_demo"))
    parser.add_argument("--review-dir", type=Path, default=Path("outputs/sulfide_qa_reviews"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Sulfide QA", layout="wide")
    st.title("Sulfide QA")
    runs = discover_runs(args.runs_dir)
    if not runs:
        st.info(f"No pipeline or inference summaries found under {args.runs_dir}")
        return

    labels = [run["label"] for run in runs]
    selected_label = st.sidebar.selectbox("Run", labels)
    run = runs[labels.index(selected_label)]
    review_path = args.review_dir / f"{run['id']}.json"
    existing = read_json(review_path) if review_path.exists() else {}

    st.caption(str(run["root"]))
    summary = read_json(run["summary_path"])
    binary_summary = load_optional(run.get("binary_summary"))
    ore_summary = load_optional(run.get("ore_summary"))

    top_cols = st.columns(4)
    if binary_summary:
        top_cols[0].metric("Sulfide fraction", f"{binary_summary.get('sulfide_fraction', 0.0) * 100:.1f}%")
        top_cols[1].metric("Tiles", binary_summary.get("tiles_processed", "-"))
        top_cols[2].metric("Inference seconds", binary_summary.get("seconds", "-"))
    if ore_summary:
        top_cols[3].metric("Ore class", ore_summary.get("ore_class_ru", ore_summary.get("ore_class", "-")))
        st.write(ore_summary.get("rule_text_ru", ""))

    tabs = st.tabs(["Overlays", "JSON", "Review"])
    with tabs[0]:
        show_images(run)
    with tabs[1]:
        cols = st.columns(3)
        cols[0].subheader("Pipeline")
        cols[0].json(summary)
        if binary_summary:
            cols[1].subheader("Binary")
            cols[1].json(binary_summary)
        if ore_summary:
            cols[2].subheader("Ore")
            cols[2].json(ore_summary)
    with tabs[2]:
        save_review_form(review_path, existing, run)


def discover_runs(root: Path) -> list[dict]:
    runs = []
    if not root.exists():
        return runs
    for path in sorted(root.rglob("pipeline_summary.json")):
        data = read_json(path)
        runs.append(
            {
                "id": safe_id(path.parent.relative_to(root)),
                "label": str(path.parent.relative_to(root)),
                "root": path.parent,
                "summary_path": path,
                "binary_summary": Path(data["paths"]["binary_sulfide_summary"]),
                "ore_summary": Path(data["paths"]["ore_summary"]),
                "images": pipeline_images(data["paths"]),
            }
        )
    for path in sorted(root.rglob("summary.json")):
        if path.parent.name == "ore_analysis":
            continue
        if any(run["summary_path"] == path for run in runs):
            continue
        data = read_json(path)
        paths = data.get("paths", {})
        runs.append(
            {
                "id": safe_id(path.parent.relative_to(root)),
                "label": str(path.parent.relative_to(root)),
                "root": path.parent,
                "summary_path": path,
                "binary_summary": path,
                "ore_summary": None,
                "images": {
                    "sulfide_overlay": Path(paths["overlay_preview"]) if paths.get("overlay_preview") else None,
                    "confidence": Path(paths["confidence"]) if paths.get("confidence") else None,
                    "sulfide_mask": Path(paths["sulfide_mask"]) if paths.get("sulfide_mask") else None,
                },
            }
        )
    return runs


def pipeline_images(paths: dict) -> dict:
    images = {
        "review_panel": paths.get("review_panel"),
        "source_preview": paths.get("source_preview"),
        "sulfide_overlay": paths.get("sulfide_overlay_preview"),
        "intergrowth_overlay": paths.get("intergrowth_overlay_preview"),
        "confidence_heatmap": paths.get("confidence_heatmap"),
        "confidence": paths.get("confidence"),
        "sulfide_mask": paths.get("sulfide_mask"),
    }
    return {name: Path(path) for name, path in images.items() if path}


def show_images(run: dict) -> None:
    images = run.get("images", {})
    cols = st.columns(2)
    for idx, (name, path) in enumerate(images.items()):
        if path and path.exists():
            cols[idx % 2].subheader(name.replace("_", " ").title())
            cols[idx % 2].image(str(path), use_container_width=True)


def save_review_form(review_path: Path, existing: dict, run: dict) -> None:
    statuses = ["accepted", "needs_mask_fix", "uncertain", "exclude_artifact", "bad_input"]
    status = st.selectbox("Status", statuses, index=statuses.index(existing.get("status", "accepted")))
    error_types = st.multiselect(
        "Error types",
        ["missed_sulfide", "false_sulfide", "bad_boundary", "wrong_ordinary_fine", "talc_issue", "artifact"],
        default=existing.get("error_types", []),
    )
    note = st.text_area("Note", value=existing.get("note", ""), height=120)
    if st.button("Save review", type="primary"):
        payload = {
            "schema_version": "binary-sulfide-review-v0.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run["id"],
            "run_root": str(run["root"]),
            "status": status,
            "error_types": error_types,
            "note": note,
        }
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        st.success(f"Saved {review_path}")
    if review_path.exists():
        st.subheader("Saved review")
        st.json(read_json(review_path))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional(path: Path | None) -> dict | None:
    return read_json(path) if path and path.exists() else None


def safe_id(path: Path) -> str:
    return "_".join(path.parts).replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    main()
