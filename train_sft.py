"""
train_sft.py — SFT training entry point with MLflow tracking.

Modes:
  1. Single run (best anchor config from grid search):
       python train_sft.py --model 1.5b --dataset 400
       python train_sft.py --model 3b   --dataset 4k

  2. Optuna sweep (API-Bank accuracy as objective):
       python train_sft.py --model 1.5b --dataset 400 --optuna --n-trials 20

In both modes every run is logged to MLflow:
  - params:  lr, max_length, epochs, batch_size, warmup_steps_ratio, weight_decay, model, dataset
  - metrics: val_loss (final epoch), api_bank_overall, api_bank_lv1, api_bank_lv2, api_bank_lv3

Reuses fsdp_sft_trainer.py via subprocess (torchrun, 4 GPUs).
Requires one-line fix in fsdp_sft_trainer.py: train/lr(1e-3) -> train/lr_1e3  (already applied)
"""

import argparse
import json
import os
import re
import shutil
import subprocess

import mlflow

# ── paths (same as sft_sweep.sh / sft_eval.sh) ────────────────────────────────
WORKSPACE       = "/workspace"
MODELS_DIR      = "/app/models"
CHECKPOINT_ROOT = f"{WORKSPACE}/sweep_checkpoints"
RESULTS_ROOT    = f"{WORKSPACE}/sweep_results"
DATASET_ROOT    = f"{WORKSPACE}/dataset"
APIBANK_DIR     = f"{WORKSPACE}/benchmarks/API-Bank"

MODEL_PATHS = {
    "1.5b": f"{MODELS_DIR}/Qwen2.5-1.5B-Instruct",
    "3b":   f"{MODELS_DIR}/Qwen2.5-3B-Instruct",
}

DATASET_PATHS = {
    "400": {
        "train": f"{DATASET_ROOT}/rlla_sft/train.parquet",
        "val":   f"{DATASET_ROOT}/rlla_sft/val.parquet",
    },
    "4k": {
        "train": f"{DATASET_ROOT}/rlla_4k/train.parquet",
        "val":   f"{DATASET_ROOT}/rlla_4k/val.parquet",
    },
}

# Best anchor config from grid search (54-run sweep, beats paper SFT400 by +3.17%)
BEST_CONFIG = {
    "lr":                  5e-5,
    "max_length":          4096,
    "epochs":              5,
    "batch_size":          64,
    "micro_batch_size":    4,
    "warmup_steps_ratio":  0.1,
    "weight_decay":        0.01,
}


# ── training ───────────────────────────────────────────────────────────────────

def run_training(model_size, dataset_size, lr, max_length, epochs,
                 batch_size, micro_batch_size, warmup_steps_ratio,
                 weight_decay, run_name, log_file, mlflow_experiment):
    """Launch fsdp_sft_trainer via torchrun (4 GPUs). Returns (returncode, val_loss, ckpt_dir)."""

    ckpt_dir = f"{CHECKPOINT_ROOT}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={DATASET_PATHS[dataset_size]['train']}",
        f"data.val_files={DATASET_PATHS[dataset_size]['val']}",
        "data.prompt_key=prompt",
        "data.response_key=response",
        f"data.max_length={max_length}",
        "data.truncation=right",
        f"data.train_batch_size={batch_size}",
        f"data.micro_batch_size={micro_batch_size}",
        f"model.partial_pretrain={MODEL_PATHS[model_size]}",
        "model.enable_gradient_checkpointing=True",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.project_name={mlflow_experiment}",
        f"trainer.experiment_name={run_name}",
        f"trainer.total_epochs={epochs}",
        "trainer.logger=['mlflow']",
        f"optim.lr={lr}",
        f"optim.warmup_steps_ratio={warmup_steps_ratio}",
        f"optim.weight_decay={weight_decay}",
    ]

    # Pass active MLflow run_id to subprocess so trainer logs into same run
    active = mlflow.active_run()
    env = {**os.environ, 'MLFLOW_RUN_ID': active.info.run_id} if active else None

    print(f"\n[train] {run_name}  log -> {log_file}")
    with open(log_file, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    return rc, _parse_val_loss(log_file), ckpt_dir


def _parse_val_loss(log_file):
    """Parse final val/loss from training log. Returns float or None."""
    val_loss = None
    if not os.path.exists(log_file):
        return None
    with open(log_file) as f:
        for line in f:
            m = re.search(r"val/loss[:\s]+([0-9.]+)", line)
            if m:
                val_loss = float(m.group(1))  # last match = final epoch
    return val_loss


# ── checkpoint cleanup ─────────────────────────────────────────────────────────

def delete_intermediate_checkpoints(ckpt_dir):
    """Delete all but the final global_step_N subdir to save disk space."""
    subdirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("global_step_")],
                     key=lambda d: int(d.split("_")[-1]))
    for d in subdirs[:-1]:  # keep only last
        shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)
        print(f"[cleanup] deleted intermediate: {d}")

def delete_checkpoint(ckpt_dir):
    """Delete entire checkpoint dir (used in Optuna when trial is not best)."""
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"[cleanup] deleted checkpoint: {ckpt_dir}")


# ── API-Bank evaluation ────────────────────────────────────────────────────────

def run_apibank_eval(ckpt_dir):
    """Run generate_batch.py + evaluate.py. Returns metrics dict or None on failure."""
    subdirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("global_step_")])
    if not subdirs:
        print(f"[eval] no global_step dir in {ckpt_dir}")
        return None
    model_path = os.path.join(ckpt_dir, subdirs[-1])

    # Delete any stale result folder to avoid resuming from corrupt partial results
    mangled = model_path.replace("/", "_")
    stale_result_dir = os.path.join(APIBANK_DIR, "PATH_TO_YOUR_SCORE_ROOT", mangled)
    if os.path.exists(stale_result_dir):
        shutil.rmtree(stale_result_dir)
        print(f"[eval] deleted stale result folder: {stale_result_dir}")

    print(f"[eval] generate_batch.py on {model_path}")
    eval_env = {**os.environ, "WORLD_SIZE": "4"}
    rc = subprocess.run(
        ["python", "generate_batch.py", "--model_paths", model_path],
        cwd=APIBANK_DIR, env=eval_env,
    ).returncode
    if rc != 0:
        print("[eval] generate_batch.py failed")
        return None

    print("[eval] evaluate.py")
    subprocess.run(
        ["python", "evaluate.py", "--model_paths", model_path],
        cwd=APIBANK_DIR,
    )

    leaderboard_path = os.path.join(APIBANK_DIR, "PATH_TO_YOUR_SCORE_ROOT", "leaderboard.json")
    if not os.path.exists(leaderboard_path):
        print("[eval] leaderboard.json not found")
        return None

    mangled = model_path.replace("/", "_")
    scores = json.load(open(leaderboard_path)).get(mangled, {})
    if not scores:
        print(f"[eval] no scores for key: {mangled}")
        return None

    return {
        "api_bank_overall": scores.get("overall_acc", 0.0),
        "api_bank_lv1":     scores.get("lv1_acc", 0.0),
        "api_bank_lv2":     scores.get("lv2_acc", 0.0),
        "api_bank_lv3":     scores.get("lv3_acc", 0.0),
    }


# ── single run ─────────────────────────────────────────────────────────────────

def single_run(args):
    """One training job with best anchor config, logged to MLflow."""
    cfg          = BEST_CONFIG.copy()
    run_name     = f"sft_{args.model}_{args.dataset}_best"
    results_dir  = f"{RESULTS_ROOT}/{args.model}_{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)

    experiment_name = f"toolrl-sft-{args.model}-{args.dataset}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**cfg, "model": args.model, "dataset": args.dataset, "mode": "single_best"})

        rc, val_loss, ckpt_dir = run_training(
            model_size=args.model, dataset_size=args.dataset,
            run_name=run_name, log_file=f"{results_dir}/{run_name}.log",
            mlflow_experiment=experiment_name,
            **{k: cfg[k] for k in ["lr", "max_length", "epochs", "batch_size",
                                    "micro_batch_size", "warmup_steps_ratio", "weight_decay"]},
        )

        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss)

        if rc != 0:
            mlflow.set_tag("status", "train_failed")
            print(f"[ERROR] training failed (exit {rc})")
            return

        delete_intermediate_checkpoints(ckpt_dir)  # keep only final epoch

        scores = run_apibank_eval(ckpt_dir)
        if scores:
            mlflow.log_metrics(scores)
            mlflow.set_tag("status", "done")
            print(f"\n[result] API-Bank: {scores['api_bank_overall']:.2f}%")
        else:
            mlflow.set_tag("status", "eval_failed")


# ── optuna sweep ───────────────────────────────────────────────────────────────

def optuna_sweep(args):
    """Optuna sweep with API-Bank accuracy as objective. All trials logged to MLflow."""
    import optuna

    study_name  = f"sft-optuna-{args.model}-{args.dataset}"
    results_dir = f"{RESULTS_ROOT}/optuna_{args.model}_{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)

    mlflow.set_experiment(study_name)

    best_score = [0.0]   # mutable so inner function can update it
    best_ckpt  = [None]

    def objective(trial):
        is_4k = args.dataset == "4k"
        lr                 = trial.suggest_float("lr", 1e-6, 1e-4 if is_4k else 5e-4, log=True)
        max_length         = trial.suggest_categorical("max_length", [1024, 2048, 4096])
        epochs             = trial.suggest_int("epochs", 1, 6 if is_4k else 7)
        batch_size         = trial.suggest_categorical("batch_size", [16, 32, 64, 128] if is_4k else [16, 32, 64])
        warmup_steps_ratio = trial.suggest_float("warmup_steps_ratio", 0.0, 0.1)
        weight_decay       = trial.suggest_float("weight_decay", 0.0, 0.1)
        micro_batch_size   = 4 if max_length >= 4096 else 8   # avoid OOM

        run_name = f"optuna_{args.model}_{args.dataset}_trial{trial.number:03d}"
        log_file = f"{results_dir}/{run_name}.log"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "trial": trial.number, "lr": lr, "max_length": max_length,
                "epochs": epochs, "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "warmup_steps_ratio": warmup_steps_ratio,
                "weight_decay": weight_decay,
                "model": args.model, "dataset": args.dataset,
            })

            rc, val_loss, ckpt_dir = run_training(
                model_size=args.model, dataset_size=args.dataset,
                lr=lr, max_length=max_length, epochs=epochs,
                batch_size=batch_size, micro_batch_size=micro_batch_size,
                warmup_steps_ratio=warmup_steps_ratio, weight_decay=weight_decay,
                run_name=run_name, log_file=log_file,
                mlflow_experiment=study_name,
            )

            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss)

            if rc != 0:
                mlflow.set_tag("status", "train_failed")
                return 0.0

            scores = run_apibank_eval(ckpt_dir)
            if scores is None:
                mlflow.set_tag("status", "eval_failed")
                return 0.0

            mlflow.log_metrics(scores)
            mlflow.set_tag("status", "done")
            score = scores["api_bank_overall"]
            print(f"\n[trial {trial.number}] API-Bank: {score:.2f}%  val_loss: {val_loss}")

            # Keep only the best checkpoint across all trials to save disk space
            if score > best_score[0]:
                if best_ckpt[0] is not None:
                    delete_checkpoint(best_ckpt[0])
                best_score[0] = score
                best_ckpt[0]  = ckpt_dir
            else:
                delete_checkpoint(ckpt_dir)

            return score

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{results_dir}/optuna.db",
        load_if_exists=True,   # resume safely if interrupted
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial {best.number}: API-Bank={best.value:.2f}%")
    print(f"Best params: {best.params}")
    print(f"Best checkpoint: {best_ckpt[0]}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT training with MLflow tracking")
    parser.add_argument("--model",    required=True, choices=["1.5b", "3b"],
                        help="Model size")
    parser.add_argument("--dataset",  required=True, choices=["400", "4k"],
                        help="Dataset: 400=rlla_sft (380 samples), 4k=rlla_4k (3980 samples)")
    parser.add_argument("--optuna",   action="store_true",
                        help="Run Optuna sweep instead of single best-config run")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20, only used with --optuna)")
    args = parser.parse_args()

    if args.optuna:
        optuna_sweep(args)
    else:
        single_run(args)


if __name__ == "__main__":
    main()
