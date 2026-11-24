"""Checkpoint and provenance helpers for the separate PCRE experiment trainer."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.generic,)):
        return json_value(value.item())
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def append_json(path, payload):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(payload), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def atomic_save(payload, destination, reserve_bytes=2 * 1024 ** 3):
    """Publish a completed, immutable checkpoint; refuse existing destinations."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(v) for v in value)
        return 0
    required = int(tensor_bytes(payload) * 1.05) + reserve_bytes
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError("Insufficient space for checkpoint plus the configured reserve")
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, destination)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def write_provenance(directory, args):
    directory = Path(directory)
    (directory / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
    sources = [Path("scripts/train_voc_adaptive.py"), Path("utils/adaptive_threshold.py"),
               Path("utils/pcre_experiment_state.py"), Path("engine/validation_adaptive.py"),
               Path("scripts/train_voc.py"), Path("model/model_wsddn.py"),
               Path("utils/camutils.py"), Path("utils/optimizer.py")]
    sources.extend(sorted(Path("model").rglob("*.py")))
    sources.extend(sorted(Path("datasets").rglob("*.py")))
    sources.extend(Path(args.list_folder).glob("*.txt"))
    sources.append(Path(args.list_folder) / "cls_labels_onehot.npy")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(sources)}
    (directory / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    archive = directory / "source_snapshot"
    archive.mkdir()
    for path in dict.fromkeys(sources):
        # All experiment source/list paths are repo-relative in supplied configs.
        target = archive / (path if not path.is_absolute() else Path("external") / path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for command, name in [(["git", "rev-parse", "HEAD"], "git_commit.txt"),
                          (["git", "diff", "--", "."], "existing_changes.patch")]:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        (directory / name).write_text(result.stdout + result.stderr)


def load_training_weights(checkpoint, checkpoint_path):
    """A full training state references its accompanying immutable weights."""
    if 'model' in checkpoint:
        return checkpoint['model']
    name = checkpoint['model_filename']
    if Path(name).name != name:
        raise ValueError('Expected a same-directory model filename')
    return torch.load(Path(checkpoint_path).resolve().parent / name,
                      map_location='cpu', weights_only=True)
