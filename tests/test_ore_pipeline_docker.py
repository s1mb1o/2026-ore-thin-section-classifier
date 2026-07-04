from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OrePipelineDockerArtifactsTest(unittest.TestCase):
    def test_docker_runtime_files_keep_vm_contract(self) -> None:
        dockerfile = (ROOT / "docker/ore-pipeline-ui/Dockerfile").read_text(encoding="utf-8")
        gx10_ml_dockerfile = (ROOT / "docker/ore-pipeline-ui/Dockerfile.gx10-ml").read_text(encoding="utf-8")
        entrypoint = (ROOT / "docker/ore-pipeline-ui/entrypoint.sh").read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.ore-pipeline-ui.yml").read_text(encoding="utf-8")
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("FROM python:3.11-slim", dockerfile)
        self.assertIn("ORE_UI_HOST=0.0.0.0", dockerfile)
        self.assertIn("ORE_UI_PORT=8080", dockerfile)
        self.assertIn("ORE_UI_BACKEND=heuristic", dockerfile)
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn("opencv-python-headless", (ROOT / "docker/ore-pipeline-ui/requirements.txt").read_text(encoding="utf-8"))

        self.assertIn("--workspace-dir", entrypoint)
        self.assertIn("--backend", entrypoint)
        self.assertIn("ORE_UI_CHECKPOINT", entrypoint)
        self.assertIn("ORE_UI_TALC_BACKEND", entrypoint)
        self.assertIn("ORE_UI_TALC_CHECKPOINT", entrypoint)
        self.assertIn("ORE_UI_TALC_THRESHOLD", entrypoint)
        self.assertIn("ORE_UI_GRADE_CHECKPOINT", entrypoint)
        self.assertIn("mkdir -p", entrypoint)

        self.assertIn("${ORE_UI_PUBLIC_PORT:-8080}:8080", compose)
        self.assertIn("./outputs/ore_pipeline_ui:/data/ore_pipeline_ui", compose)
        self.assertIn("./models:/app/models:ro", compose)
        self.assertIn("ORE_UI_TALC_BACKEND", compose)
        self.assertIn("ORE_UI_GRADE_CHECKPOINT", compose)
        self.assertIn("restart: unless-stopped", compose)

        self.assertIn("FROM nvcr.io/nvidia/pytorch:25.11-py3", gx10_ml_dockerfile)
        self.assertIn("ORE_UI_BACKEND=ml", gx10_ml_dockerfile)
        self.assertIn("ORE_UI_TALC_BACKEND=ml", gx10_ml_dockerfile)
        self.assertIn("requirements-gx10-ml.txt", gx10_ml_dockerfile)
        self.assertIn("ENTRYPOINT", gx10_ml_dockerfile)
        self.assertIn("transformers", (ROOT / "docker/ore-pipeline-ui/requirements-gx10-ml.txt").read_text(encoding="utf-8"))

        self.assertRegex(dockerignore, r"(?m)^dataset$")
        self.assertRegex(dockerignore, r"(?m)^outputs/\*$")
        self.assertRegex(dockerignore, r"(?m)^models/\*$")
        self.assertIn("*.safetensors", dockerignore)


if __name__ == "__main__":
    unittest.main()
