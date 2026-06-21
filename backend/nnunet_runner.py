"""
Run nnU-Net v2 inference via the `nnUNetv2_predict` CLI.

Supports an optional uploaded checkpoint: it's placed into a temporary model
folder (alongside the trained plans.json/dataset.json) and used via `-chk`, so
the on-disk model folder is never modified.

Env vars (see .env.example):
  nnUNet_results / nnUNet_raw / nnUNet_preprocessed
  NNUNET_DATASET, NNUNET_CONFIG, NNUNET_FOLDS, NNUNET_TRAINER, NNUNET_PLANS,
  NNUNET_CHECKPOINT, NNUNET_DEVICE, NNUNET_DISABLE_TTA
"""
import glob
import os
import shutil
import subprocess
import tempfile


def _resolve_dataset_folder(results_dir, dataset):
    if dataset.startswith("Dataset"):
        cand = os.path.join(results_dir, dataset)
        if os.path.isdir(cand):
            return cand
        raise RuntimeError(f"Model folder not found: {cand}")
    try:
        did = int(dataset)
    except ValueError:
        raise RuntimeError(f"NNUNET_DATASET must be an id or DatasetXXX_Name (got {dataset!r})")
    matches = sorted(glob.glob(os.path.join(results_dir, f"Dataset{did:03d}_*")))
    if not matches:
        raise RuntimeError(f"No Dataset{did:03d}_* folder under {results_dir}")
    return matches[0]


def _model_dir(results_dir, dataset, trainer, plans, config):
    ds_dir = _resolve_dataset_folder(results_dir, dataset)
    model_dir = os.path.join(ds_dir, f"{trainer}__{plans}__{config}")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Configuration folder not found: {model_dir}")
    return ds_dir, model_dir


def run_nnunet(input_nifti_path, checkpoint_path=None):
    results_dir = os.environ.get("nnUNet_results", "")
    dataset = os.environ.get("NNUNET_DATASET", "110")
    config = os.environ.get("NNUNET_CONFIG", "3d_fullres")
    folds = os.environ.get("NNUNET_FOLDS", "0").split()
    trainer = os.environ.get("NNUNET_TRAINER", "nnUNetTrainer")
    plans = os.environ.get("NNUNET_PLANS", "nnUNetPlans")
    device = os.environ.get("NNUNET_DEVICE", "cpu")
    disable_tta = os.environ.get("NNUNET_DISABLE_TTA", "1") == "1"
    default_chk = os.environ.get("NNUNET_CHECKPOINT", "checkpoint_final.pth")

    env = os.environ.copy()
    work = tempfile.mkdtemp(prefix="nnunet_")
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    shutil.copy(input_nifti_path, os.path.join(in_dir, "case_0000.nii.gz"))

    try:
        if checkpoint_path:
            ds_dir, model_dir = _model_dir(results_dir, dataset, trainer, plans, config)
            for req in ("plans.json", "dataset.json"):
                if not os.path.isfile(os.path.join(model_dir, req)):
                    raise RuntimeError(
                        f"{req} is missing from {model_dir}. nnU-Net needs "
                        "plans.json and dataset.json from your trained model to "
                        "use a checkpoint. Copy them into the model folder once."
                    )
            tmp_results = os.path.join(work, "results")
            tmp_model = os.path.join(tmp_results, os.path.basename(ds_dir),
                                     os.path.basename(model_dir))
            os.makedirs(tmp_model)
            for fn in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
                src = os.path.join(model_dir, fn)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(tmp_model, fn))
            chk_name = os.path.basename(checkpoint_path)
            if not chk_name.endswith(".pth"):
                chk_name += ".pth"
            for f in folds:
                fold_dir = os.path.join(tmp_model, f"fold_{f}")
                os.makedirs(fold_dir, exist_ok=True)
                shutil.copy(checkpoint_path, os.path.join(fold_dir, chk_name))
            env["nnUNet_results"] = tmp_results
            checkpoint = chk_name
        else:
            checkpoint = default_chk

        cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir,
               "-d", dataset, "-tr", trainer, "-p", plans,
               "-c", config, "-f", *folds, "-chk", checkpoint, "-device", device]
        if disable_tta:
            cmd.append("--disable_tta")

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("nnUNetv2_predict failed:\n"
                               + (proc.stderr or proc.stdout or "")[-2000:])
        segs = sorted(glob.glob(os.path.join(out_dir, "*.nii.gz")))
        if not segs:
            raise RuntimeError("nnU-Net produced no output segmentation.")
        final = input_nifti_path + ".seg.nii.gz"
        shutil.copy(segs[0], final)
        return final
    finally:
        shutil.rmtree(work, ignore_errors=True)
