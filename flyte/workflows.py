from flytekit import task, workflow, Resources, Secret, ImageSpec
import subprocess
import sys
import glob
import os
import pathlib
import random

import numpy as np
import pandas as pd

from typing import Optional

custom_image = ImageSpec(
    base_image="579102688835.dkr.ecr.us-east-1.amazonaws.com/ssi/flyte:dev-dist-latest",
    name="ssi",
    registry="579102688835.dkr.ecr.us-east-1.amazonaws.com",
    python_version="3.11",
    requirements="uv.lock",
    pip_secret_mounts=[(os.path.join(pathlib.Path.home(), ".netrc"), "/root/.netrc")],
    commands=[
        "uv pip install --system -r requirements.txt"
    ],
)

@task(
    container_image=custom_image,
    requests=Resources(cpu="4", mem="4Gi"),
    environment={
        "WANDB_API_KEY": "op://Cloud Native/wandb/token",
        "OPENAI_API_KEY": "op://Cloud Native/openai-ml-token/password",
        "ANTHROPIC_API_KEY": "op://Cloud Native/Anthropic Key/credential",
        "OPENWEBUI_API_KEY": "op://Cloud Native/openwebui-token/password",
    },
)
def run_arxiv_processor(workers: int = 1, model: str = "gpt-4.1", no_download: bool = False, limit: Optional[int] = None) -> str:
    # Build the command to mimic CLI invocation
    cmd = [sys.executable, "arxiv_processor.py"]
    if no_download:
        cmd.append("--no-download")
    if limit is not None:
        cmd.append("--limit")
        cmd.append(str(limit))
    cmd.append("--model")
    cmd.append(model)
    cmd.append("--workers")
    cmd.append(str(workers))
    
    # Run the subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Script failed: {result.stderr}")
    # After running subprocess, return output path
    output_path = "output/papers/initial_QA_pairs/all_papers.json"
    return f"Script completed. Aggregated output: {output_path}\n{result.stdout}"

@workflow
def arxiv_processor_workflow(workers: int = 1, model: str = "gpt-4.1", no_download: bool = False, limit: Optional[int] = None) -> str:
    return run_arxiv_processor(workers=workers, model=model, no_download=no_download, limit=limit) 