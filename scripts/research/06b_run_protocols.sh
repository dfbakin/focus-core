#!/usr/bin/env bash
# GPU runner for the leak-gradient experiment (docs/research/11_gpu_experiment_plan.md, sections 4 to 6).
# One GPU, strictly sequential, on the rented host only (never on the workstation: its GPU crashes the machine;
# the script refuses to train when torch sees no CUDA device, so CUDA_VISIBLE_DEVICES="" makes it abort).
#
# Blocks, in run order (cheap first, so a budget cut costs only the temporal model):
#   probe    scripts/research/06e_probe.py once: frozen ImageNet ResNet50 centre-frame features of all clips (GPU) and
#            logistic-regression fits on CPU under P0, P1 f0-4, P2 f0-4, grp27a, grp27b, grp17
#   r50_p1   ResNet50 single frame on data/p1_f{0..4}, seed 42                               (5 runs)
#   r50_p2   ResNet50 single frame on data/p2_f{0..4}, seeds 42, 43, 44 (seed-major order)  (15 runs)
#   slow_p2  SLOW 8x8 R50 on data/p2_f{0..4}, seed 42, up to 15 epochs                      (5 runs)
# The roots come from scripts/research/06_make_protocol_roots.py; the clips from `dvc pull data/ouc-cge-320.dvc`.
#
# Recipes: exactly R50_ARGS and SLOW_ARGS of scripts/review/08_run_grouped_training.sh (SLOW with
# trainer.max_epochs=15), plus experiment.seed=<seed> and trainer.checkpoint.save_top_k=-1, so that every epoch is
# saved and last.ckpt is the true final epoch (with top-k = 2, as in 08, last.ckpt was the later of the two best
# epochs; see the note in 08 and 11_epoch_curve.sh).
#
# Per training run <run> = p{1|2}_f{k}_{r50|slow}_s{seed}:
#   training      python -m src.train ... paths.output_dir=outputs/research_gpu/<run>
#                 mlflow.experiment_name=research_protocols; log $LOGS/train_<run>.log; abort the script on failure
#   evaluation    scripts/review/05_eval_checkpoints.py (the project's test_step, per-clip predictions):
#                   best-by-validation checkpoint on val and test       tags <run>_best_val, <run>_best_test
#                   with the epoch curve (EPOCH_CURVE, default: the seed-42 runs, i.e. r50 p1, r50 p2 s42, slow p2):
#                   every epoch checkpoint on test (EVAL_JOBS parallel) tags <run>_ep<NN>_test (highest NN = final epoch)
#                   without it (default: seeds 43 and 44): the final epoch only on test, tag <run>_ep<NN>_test
#                 -> $OUT/eval/preds_<tag>.csv (path,label,pred,p0,p1,p2) and $OUT/eval/summary_<tag>.json
#                 best = MLflow best_checkpoint_path (Lightning: strictly better val/accuracy, earliest epoch on ties),
#                 falling back to the highest val_acc in the file names; written to $OUT/state/ckpts_<run>.json
#   cleanup       only when every evaluation above exists: seed 42 keeps the best-by-val and final-epoch files and
#                 last.ckpt (a second save of the final epoch, rsync it with --exclude last.ckpt if disk or bandwidth
#                 matter); seeds 43 and 44 delete all checkpoints
#   markers       $OUT/state/trained_<run> after training, $OUT/state/done_<run> after evaluation and cleanup
# Resume: a run with done_<run> is skipped (08's skip-if-evaluated, keyed on the marker so that an interrupted epoch
# evaluation is finished, not skipped); a run with trained_<run> but no done_<run> resumes at the evaluation.
# The probe is skipped when $OUT/state/done_probe exists.
#
# The functions code_version, need_root, post_run and run are copied from 08_run_grouped_training.sh, so that the
# new runs are trained, located in MLflow, evaluated and logged exactly as the October reference runs R1-R5; the
# changes are the paths, the tags, the per-epoch evaluation of 11_epoch_curve.sh (eval_one), the checkpoint cleanup,
# the resume markers and the DRY mode.
#
# Usage (from anywhere; the script changes to the repository root):
#   VENV=/venv/main nohup bash scripts/research/06b_run_protocols.sh > research_gpu.log 2>&1 &
#   DRY=1 bash scripts/research/06b_run_protocols.sh                   # print every command, check the roots, run nothing
#   RUNS="probe r50_p1" bash scripts/research/06b_run_protocols.sh     # blocks
#   RUNS="p2_f4_r50_s42 p2_f4_slow_s42" bash scripts/research/06b_run_protocols.sh   # single runs
#   BUDGET_MIN=200 bash scripts/research/06b_run_protocols.sh
# Environment:
#   VENV          virtualenv to activate first (optional; /venv/main on the vast.ai image)
#   OUT           summaries                         (default outputs/research_gpu/_summary: eval/ mlflow/ logs/ state/ runs.log)
#   LOGS          per-run logs                      (default $OUT/logs)
#   RUN_BASE      per-run output dirs               (default outputs/research_gpu)
#   RUNS          space-separated run names and/or block names probe r50_p1 r50_p2 slow_p2 (default: all)
#   EVAL_JOBS     parallel checkpoint evaluations   (default 2)
#   EPOCH_CURVE   which runs get every epoch evaluated on test: seed42 (default), all, none; the others get the
#                 best-by-val checkpoint on val and test and the final epoch on test only
#   PROBE_WORKERS decode processes of the probe     (default 16)
#   FORCE=1       retrain and re-evaluate runs that are done (their old checkpoints and predictions are deleted)
#   DRY=1         print every command and check that every needed root exists; nothing runs, nothing is written
#   BUDGET_MIN    wall-clock budget in minutes (default 0 = none). Before each seed group of r50_p2 and before slow_p2
#                 the remaining work is projected (measured mean wall time per recipe and evaluation mode, else the
#                 estimates in minutes per run including evaluation: R50_EST_MIN=8 and SLOW_EST_MIN=18 with the epoch
#                 curve, R50_NOCURVE_EST_MIN=7 and SLOW_NOCURVE_EST_MIN=13 without it; PROBE_EST_MIN=10) and, while it does not fit,
#                 units are dropped in this order: slow_p2, then r50_p2 seed 44, then r50_p2 seed 43.  With DRY=1 the
#                 elapsed time is the sum of the estimates of the runs before the decision point.  The decision is
#                 recomputed at every point for the units still ahead, from the measured times
# Output: per-run dirs outputs/research_gpu/<run>/{checkpoints,hydra,config.yaml}; ./mlruns (MLflow file store);
# summaries under $OUT; the last line is ALL_DONE with the total wall time.  Aggregate with
#   python scripts/research/06c_aggregate.py
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
# MLflow >= 3.17 refuses the file-store backend that the project's `mlflow.tracking_uri: null` resolves to unless set.
export MLFLOW_ALLOW_FILE_STORE=true
OUT=${OUT:-outputs/research_gpu/_summary}
LOGS=${LOGS:-$OUT/logs}
RUN_BASE=${RUN_BASE:-outputs/research_gpu}
EVAL_JOBS=${EVAL_JOBS:-2}
PROBE_WORKERS=${PROBE_WORKERS:-16}
DRY=${DRY:-0}
FORCE=${FORCE:-0}
BUDGET_MIN=${BUDGET_MIN:-0}
EPOCH_CURVE=${EPOCH_CURVE:-seed42}
R50_EST_MIN=${R50_EST_MIN:-8}
SLOW_EST_MIN=${SLOW_EST_MIN:-18}
R50_NOCURVE_EST_MIN=${R50_NOCURVE_EST_MIN:-7}
SLOW_NOCURVE_EST_MIN=${SLOW_NOCURVE_EST_MIN:-13}
PROBE_EST_MIN=${PROBE_EST_MIN:-10}
VCLOCK=0   # DRY mode: virtual elapsed seconds (sum of the estimates of the runs printed so far) for the budget plan
EXPERIMENT=research_protocols
T_START=$(date +%s)
export OUT LOGS FORCE

R50_ARGS="+experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16"
SLOW_ARGS="+experiment=baseline_ouc_cge trainer.early_stopping.patience=5 trainer.early_stopping.min_epochs=5 data.batch_size=32 data.num_workers=16 trainer.precision=bf16-mixed trainer.max_epochs=15"
EXTRA_ARGS="trainer.checkpoint.save_top_k=-1"
FOLDS="0 1 2 3 4"
P2_SEEDS="42 43 44"

log() { if [ "$DRY" = "1" ]; then echo "$*"; else echo "$*" | tee -a "$OUT/runs.log"; fi; }

case "$EPOCH_CURVE" in seed42|all|none) ;; *) echo "ABORT: EPOCH_CURVE must be seed42, all or none (got '$EPOCH_CURVE')"; exit 4 ;; esac
curve_for() {  # curve_for <seed> : 1 when this run gets every epoch evaluated on test
  case "$EPOCH_CURVE" in all) echo 1 ;; none) echo 0 ;; *) if [ "$1" = "42" ]; then echo 1; else echo 0; fi ;; esac
}
est_sec() {  # est_sec <model> <curve> : a-priori seconds per run including evaluation
  case "$1:$2" in
    slow:1) echo $((SLOW_EST_MIN * 60)) ;; slow:0) echo $((SLOW_NOCURVE_EST_MIN * 60)) ;;
    *:1)    echo $((R50_EST_MIN * 60)) ;;  *)      echo $((R50_NOCURVE_EST_MIN * 60)) ;;
  esac
}
declare -A WSUM WN   # measured wall time (with evaluation) per "<model>:<curve>" of the runs trained in this call
per_run_sec() {  # per_run_sec <model> <curve> : measured mean, else the estimate
  local k="$1:$2"
  if [ "${WN[$k]:-0}" -gt 0 ]; then echo $(( ${WSUM[$k]} / ${WN[$k]} )); else est_sec "$1" "$2"; fi
}

if [ "$DRY" != "1" ]; then
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ABORT: torch sees no CUDA device (or python/torch is missing). This runner trains on the rented GPU host only."
    exit 3
  fi
  mkdir -p "$OUT/eval" "$OUT/mlflow" "$OUT/state" "$LOGS"
fi

code_version() {  # copied from 08
  # Commit of the checkout, marked when src/ or configs/ carry uncommitted changes.
  local rev
  rev=$(git rev-parse --short HEAD 2>/dev/null) || { echo "no-git"; return; }
  if git diff --quiet HEAD -- src/ configs/ 2>/dev/null; then echo "$rev"; else echo "$rev+local-changes"; fi
}

want() {  # want <run name> <block name>
  [ -z "${RUNS:-}" ] || [[ " ${RUNS} " == *" $1 "* ]] || [[ " ${RUNS} " == *" $2 "* ]]
}

MISSING_ROOTS=""
need_root() {  # need_root <root> : the dataset root must exist before a run starts (copied from 08; DRY collects)
  local root=$1
  if [ ! -f "$root/data_labels/train.csv" ] || [ ! -f "$root/data_labels/val.csv" ] || \
     [ ! -f "$root/data_labels/test.csv" ] || [ ! -d "$root/videos/" ]; then
    if [ "$DRY" = "1" ]; then
      [[ " $MISSING_ROOTS " == *" $root "* ]] || MISSING_ROOTS="$MISSING_ROOTS $root"
      return 1
    fi
    log "MISSING dataset root $root (data_labels/{train,val,test}.csv and videos/)."
    log "  clips:  dvc pull data/ouc-cge-320.dvc"
    log "  roots:  python scripts/research/06_make_protocol_roots.py"
    exit 2
  fi
  return 0
}

need_file() {  # need_file <path> : an input file of the probe
  if [ ! -e "$1" ]; then
    if [ "$DRY" = "1" ]; then MISSING_ROOTS="$MISSING_ROOTS $1"; return 1; fi
    log "MISSING $1"; exit 2
  fi
}

elapsed_min() { echo $(( ($(date +%s) - T_START) / 60 )); }

eval_one() {  # eval_one <ckpt> <model> <root> <name> : one epoch checkpoint on test (from 11_epoch_curve.sh)
  local ckpt=$1 model=$2 root=$3 name=$4 ep tag
  ep=$(basename "$ckpt" | sed -E 's/^epoch=([0-9]+)-.*/\1/')
  tag="${name}_ep${ep}_test"
  if [ -f "$OUT/eval/summary_${tag}.json" ]; then echo "  $name epoch $ep already evaluated"; return 0; fi
  python scripts/review/05_eval_checkpoints.py --model "$model" --ckpt "$ckpt" --root "$root" --split test \
      --mode identity --out "$OUT/eval" --tag "$tag" > "$LOGS/eval_${tag}.log" 2>&1
  echo "  $name epoch $ep rc=$? $(grep -o '"lightning_test_accuracy": [0-9.]*' "$OUT/eval/summary_${tag}.json" 2>/dev/null)"
}
export -f eval_one

post_run() {  # post_run <name> <model_key> <root> <seed> <curve> : MLflow copy, checkpoint choice, evaluations, cleanup
  local name=$1 model=$2 root=$3 seed=$4 curve=$5 rdir="$RUN_BASE/$1"
  python - "$name" "$OUT" "$rdir" <<'EOF'
import json, shutil, sys
from pathlib import Path
name, out, want = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
hits = [p.parents[1] for p in Path("mlruns").glob("*/*/params/output_dir") if p.read_text().strip() == want]
if not hits:
    print("MLFLOW_RUN_NOT_FOUND"); sys.exit(0)
hit = max(hits, key=lambda d: (d / "meta.yaml").stat().st_mtime)  # the latest run when a run was repeated
dst = out / "mlflow" / name
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(hit / "metrics", dst / "metrics"); shutil.copytree(hit / "params", dst / "params")
if (hit / "tags").exists(): shutil.copytree(hit / "tags", dst / "tags")
summary = {"run_id": hit.name, "experiment_id": hit.parent.name}
for f in sorted((hit / "metrics").glob("**/*")):
    if f.is_file():
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        key = str(f.relative_to(hit / "metrics"))
        summary[key] = {"last": float(rows[-1][1]), "last_step": int(rows[-1][2]), "n": len(rows),
                        "max": max(float(r[1]) for r in rows), "min": min(float(r[1]) for r in rows)}
for f in (hit / "params").glob("*"):
    summary["param_" + f.name] = f.read_text().strip()
json.dump(summary, open(dst / "summary.json", "w"), indent=2)
print(f"MLFLOW run={hit.name} best_val={summary.get('best_val/accuracy', {}).get('last')} best_epoch={summary.get('best_epoch', {}).get('last')} "
      f"final_epoch={summary.get('final_epoch', {}).get('last')} test_acc={summary.get('test/accuracy', {}).get('last')} "
      f"test_f1={summary.get('test/f1_macro', {}).get('last')} train_acc_last={summary.get('train/accuracy', {}).get('last')}")
EOF
  # best-by-validation and final-epoch checkpoints (replaces 08's ls | sort fallback)
  local sel best final
  sel=$(python - "$OUT/mlflow/$name/summary.json" "$rdir/checkpoints" "$OUT/state/ckpts_${name}.json" <<'EOF'
import json, re, sys
from pathlib import Path
summ, cdir, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
eps = []
for p in sorted(cdir.glob("epoch=*.ckpt")):
    m = re.match(r"epoch=(\d+)-val_acc=([0-9.]+?)(-v\d+)?\.ckpt$", p.name)
    if m: eps.append((int(m.group(1)), float(m.group(2)), str(p)))
if not eps:
    print("NONE NONE"); sys.exit(0)
by_name = max(eps, key=lambda e: (e[1], -e[0]))          # highest val_acc, earliest epoch on ties
mlf = ""
try: mlf = json.load(open(summ)).get("param_best_checkpoint_path", "")
except Exception: pass
hit = [e for e in eps if mlf and Path(e[2]).name == Path(mlf).name]
best, source = (hit[0], "mlflow best_checkpoint_path") if hit else (by_name, "highest val_acc in file names")
final = max(eps)
info = dict(best_ckpt=best[2], best_epoch=best[0], best_val_acc=best[1], best_source=source,
            best_by_filename_epoch=by_name[0], final_ckpt=final[2], final_epoch=final[0], n_epochs=len(eps),
            epochs={f"{e[0]:02d}": e[1] for e in eps}, mlflow_best_checkpoint_path=mlf)
json.dump(info, open(dst, "w"), indent=1)
if hit and by_name[0] != best[0]:
    print(f"NOTE mlflow best epoch {best[0]} != file-name best epoch {by_name[0]} (rounded val_acc)", file=sys.stderr)
print(best[2], final[2])
EOF
)
  best=${sel% *}; final=${sel#* }
  if [ "$best" = "NONE" ]; then echo "NO EPOCH CHECKPOINTS in $rdir/checkpoints; evaluation impossible"; return 0; fi
  echo "eval $name best=$best final=$final"
  local split tag
  for split in test val; do
    tag="${name}_best_${split}"
    if [ -f "$OUT/eval/summary_${tag}.json" ]; then echo "  best $split already evaluated"; continue; fi
    python scripts/review/05_eval_checkpoints.py --model "$model" --ckpt "$best" --root "$root" --split $split \
        --mode identity --out "$OUT/eval" --tag "$tag" > "$LOGS/eval_${tag}.log" 2>&1
    echo "  eval best $split rc=$? $(grep -o '"lightning_test_accuracy": [0-9.]*' "$OUT/eval/summary_${tag}.json" 2>/dev/null)"
  done
  local expected=()  # epoch checkpoints whose test evaluation must exist before any checkpoint is deleted
  if [ "$curve" = "1" ]; then
    expected=("$rdir/checkpoints/"epoch=*.ckpt)
    ls "$rdir/checkpoints/"epoch=*.ckpt | \
        xargs -P "$EVAL_JOBS" -I{} bash -c 'eval_one "$@"' _ {} "$model" "$root" "$name" | sort
  else
    expected=("$final")
    echo "  epoch curve off (EPOCH_CURVE=$EPOCH_CURVE, seed $seed): final epoch only"
    eval_one "$final" "$model" "$root" "$name"
  fi
  echo "  Skipping-count train log: $(grep -c Skipping "$LOGS/train_$name.log" 2>/dev/null)"
  # completeness: every expected summary must exist before checkpoints are deleted
  local missing=0 f ep
  for f in "${name}_best_test" "${name}_best_val"; do [ -f "$OUT/eval/summary_$f.json" ] || missing=$((missing + 1)); done
  for f in "${expected[@]}"; do
    ep=$(basename "$f" | sed -E 's/^epoch=([0-9]+)-.*/\1/')
    [ -f "$OUT/eval/summary_${name}_ep${ep}_test.json" ] || missing=$((missing + 1))
  done
  if [ $missing -gt 0 ]; then
    echo "EVAL INCOMPLETE $name: $missing evaluation(s) missing; checkpoints kept, rerun the script to finish"
    return 0
  fi
  if [ "$seed" = "42" ]; then
    local keep_b keep_f
    keep_b=$(basename "$best"); keep_f=$(basename "$final")
    for f in "$rdir/checkpoints/"epoch=*.ckpt; do
      if [ "$(basename "$f")" != "$keep_b" ] && [ "$(basename "$f")" != "$keep_f" ]; then rm -f "$f"; fi
    done
    echo "  kept $(ls "$rdir/checkpoints/" | tr '\n' ' ')($(du -sh "$rdir/checkpoints" | cut -f1))"
  else
    rm -f "$rdir/checkpoints/"*.ckpt
    echo "  seed $seed: all checkpoints deleted after evaluation"
  fi
  date -Is > "$OUT/state/done_${name}"
}

INCOMPLETE=""
run() {  # run <name> <model_key> <root> <seed> <hydra overrides...>   (copied from 08, adapted)
  local name=$1 model=$2 root=$3 seed=$4; shift 4
  local rdir="$RUN_BASE/$name" curve
  curve=$(curve_for "$seed")
  if [ -f "$OUT/state/done_${name}" ] && [ "$FORCE" != "1" ]; then
    log "=== $name already evaluated, skipping (FORCE=1 to redo)"; return 0
  fi
  need_root "$root" || true
  local cmd=(python -m src.train "$@" experiment.seed="$seed" $EXTRA_ARGS data.root="$root" paths.output_dir="$rdir"
             mlflow.experiment_name=$EXPERIMENT hydra.run.dir="$rdir/hydra")
  if [ "$DRY" = "1" ]; then
    echo "=== $name model=$model root=$root seed=$seed"
    echo "CMD: ${cmd[*]} > $LOGS/train_$name.log"
    echo "EVAL: python scripts/review/05_eval_checkpoints.py --model $model --ckpt <best-by-val> --root $root --split {test,val} --mode identity --out $OUT/eval --tag ${name}_best_{test,val}"
    if [ "$curve" = "1" ]; then
      echo "EVAL: for each $rdir/checkpoints/epoch=NN-*.ckpt (xargs -P $EVAL_JOBS): ... --split test --tag ${name}_epNN_test"
    else
      echo "EVAL: final epoch only: ... --ckpt $rdir/checkpoints/epoch=<final>-*.ckpt --split test --tag ${name}_ep<final>_test"
    fi
    if [ "$seed" = "42" ]; then echo "CLEANUP: keep best-by-val, final epoch, last.ckpt"; else echo "CLEANUP: rm $rdir/checkpoints/*.ckpt"; fi
    VCLOCK=$((VCLOCK + $(est_sec "$model" "$curve")))
    return 0
  fi
  local t0=$(date +%s) cv trained=0
  cv=$(code_version)
  if [ -f "$OUT/state/trained_${name}" ] && [ "$FORCE" != "1" ] && compgen -G "$rdir/checkpoints/epoch=*.ckpt" > /dev/null; then
    log "=== $name trained earlier (marker $OUT/state/trained_${name}); resuming at the evaluation"
  else
    # a fresh training: stale checkpoints would mix with the new epochs, stale predictions with the new evaluation
    rm -rf "$rdir/checkpoints"
    rm -f "$OUT/eval/"{preds,summary}_"${name}"_best_{test,val}.{csv,json} "$OUT/eval/"{preds,summary}_"${name}"_ep[0-9]*_test.{csv,json}
    rm -f "$OUT/state/trained_${name}" "$OUT/state/done_${name}" "$OUT/state/ckpts_${name}.json"
    log "=== $name model=$model root=$root seed=$seed code=$cv start=$(date -Is) elapsed=$(elapsed_min)min"
    log "CMD: ${cmd[*]}"
    "${cmd[@]}" > "$LOGS/train_$name.log" 2>&1
    local rc=$? t1=$(date +%s)
    log "$name rc=$rc wall=$((t1 - t0))s code=$cv"
    if [ $rc -ne 0 ]; then
      echo "TRAINING FAILED ($name); last 40 log lines:"; tail -40 "$LOGS/train_$name.log"
      log "ABORT after $name"; exit $rc
    fi
    date -Is > "$OUT/state/trained_${name}"
    trained=1
  fi
  post_run "$name" "$model" "$root" "$seed" "$curve" 2>&1 | tee -a "$OUT/runs.log"
  local wall=$(( $(date +%s) - t0 ))
  log "$name wall_with_eval=${wall}s"
  if [ ! -f "$OUT/state/done_${name}" ]; then INCOMPLETE="$INCOMPLETE $name"; fi
  if [ $trained -eq 1 ]; then
    WSUM[$model:$curve]=$(( ${WSUM[$model:$curve]:-0} + wall )); WN[$model:$curve]=$(( ${WN[$model:$curve]:-0} + 1 ))
  fi
}

run_probe() {
  if [ -f "$OUT/state/done_probe" ] && [ "$FORCE" != "1" ]; then log "=== probe already done, skipping (FORCE=1 to redo)"; return 0; fi
  need_root data/ouc-cge-320 || true
  local k p name
  for p in p1 p2; do for k in $FOLDS; do need_root "data/${p}_f$k" || true; done; done
  for name in grp27a grp27b grp17; do need_file "docs/review/data/grouped/splits/split_${name}.csv" || true; done
  local force_args=""
  if [ "$FORCE" = "1" ]; then force_args="--force-extract"; fi
  local cmd=(python scripts/research/06e_probe.py --data-root data/ouc-cge-320 --features "$OUT/probe_features.npz"
             --out-eval "$OUT/eval" --workers "$PROBE_WORKERS" $force_args)
  if [ "$DRY" = "1" ]; then
    echo "=== probe"; echo "CMD: ${cmd[*]} > $LOGS/probe.log"; VCLOCK=$((VCLOCK + PROBE_EST_MIN * 60)); return 0
  fi
  local t0=$(date +%s)
  log "=== probe code=$(code_version) start=$(date -Is) elapsed=$(elapsed_min)min"
  log "CMD: ${cmd[*]}"
  "${cmd[@]}" > "$LOGS/probe.log" 2>&1
  local rc=$?
  log "probe rc=$rc wall=$(( $(date +%s) - t0 ))s"
  if [ $rc -ne 0 ]; then
    # the probe is cheap and can be redone alone (RUNS=probe); a failure must not cost the GPU training blocks
    echo "PROBE FAILED; last 40 log lines:"; tail -40 "$LOGS/probe.log"
    log "probe FAILED, continuing with the training blocks"; INCOMPLETE="$INCOMPLETE probe"
  else
    grep -E "^(fit|features|extract)" "$LOGS/probe.log" | tee -a "$OUT/runs.log"
    date -Is > "$OUT/state/done_probe"
  fi
}

# ---------------------------------------------------------------- budget
DROPPED=""
is_dropped() { [[ " $DROPPED " == *" $1 "* ]]; }
unit_runs() {  # unit_runs <unit> : run names of a budget unit
  local k
  case $1 in
    r50_p2_s*) for k in $FOLDS; do echo "p2_f${k}_r50_s${1#r50_p2_s}"; done ;;
    slow_p2)   for k in $FOLDS; do echo "p2_f${k}_slow_s42"; done ;;
  esac
}
unit_block() { case $1 in r50_p2_s*) echo r50_p2 ;; slow_p2) echo slow_p2 ;; esac; }
unit_cost() {  # seconds still needed for a unit (runs wanted and not done)
  local u=$1 per n=0 r
  if [ "$u" = "slow_p2" ]; then
    per=$(per_run_sec slow "$(curve_for 42)")
  else
    per=$(per_run_sec resnet50 "$(curve_for "${u#r50_p2_s}")")
  fi
  for r in $(unit_runs "$u"); do
    if want "$r" "$(unit_block "$u")" && [ ! -f "$OUT/state/done_$r" -o "$FORCE" = "1" ]; then n=$((n + 1)); fi
  done
  echo $((n * per))
}
budget_check() {  # budget_check <units still ahead, in run order> : drop slow_p2, then seed 44, then seed 43
  # The drop set of the units ahead is recomputed at every decision point from the latest measured run times, so a
  # unit dropped on the a-priori estimates comes back when the measured runs are faster.
  [ "$BUDGET_MIN" -gt 0 ] || return 0
  local u cost proj d elapsed keep=""
  for u in $DROPPED; do [[ " $* " == *" $u "* ]] || keep="$keep $u"; done
  DROPPED=$keep
  while true; do
    if [ "$DRY" = "1" ]; then elapsed=$VCLOCK; else elapsed=$(( $(date +%s) - T_START )); fi
    cost=0
    for u in "$@"; do is_dropped "$u" || cost=$((cost + $(unit_cost "$u"))); done
    proj=$((elapsed + cost))
    if [ $proj -le $((BUDGET_MIN * 60)) ]; then
      log "budget: elapsed=${elapsed}s remaining_projection=${cost}s total=${proj}s budget=$((BUDGET_MIN * 60))s ok (dropped:${DROPPED:- none})"
      return 0
    fi
    d=""
    for u in slow_p2 r50_p2_s44 r50_p2_s43; do
      if [[ " $* " == *" $u "* ]] && ! is_dropped "$u"; then d=$u; break; fi
    done
    if [ -z "$d" ]; then
      log "budget: projection ${proj}s exceeds $((BUDGET_MIN * 60))s, nothing droppable is left ahead; continuing"
      return 0
    fi
    DROPPED="$DROPPED $d"
    log "budget: projection ${proj}s exceeds $((BUDGET_MIN * 60))s; dropping $d"
  done
}

# ---------------------------------------------------------------- plan
if [ "$DRY" != "1" ]; then  # fail fast: every root of every wanted run must exist before the GPU time starts
  for k in $FOLDS; do
    if want "p1_f${k}_r50_s42" r50_p1; then need_root "data/p1_f$k"; fi
    for s in $P2_SEEDS; do if want "p2_f${k}_r50_s$s" r50_p2; then need_root "data/p2_f$k"; fi; done
    if want "p2_f${k}_slow_s42" slow_p2; then need_root "data/p2_f$k"; fi
  done
  log "##### 06b_run_protocols start=$(date -Is) code=$(code_version) host=$(hostname) RUNS='${RUNS:-all}' BUDGET_MIN=$BUDGET_MIN EVAL_JOBS=$EVAL_JOBS"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /' | tee -a "$OUT/runs.log"
fi

if want probe probe; then run_probe; fi

for k in $FOLDS; do
  if want "p1_f${k}_r50_s42" r50_p1; then run "p1_f${k}_r50_s42" resnet50 "data/p1_f$k" 42 $R50_ARGS; fi
done

AHEAD="r50_p2_s42 r50_p2_s43 r50_p2_s44 slow_p2"
for s in $P2_SEEDS; do
  budget_check $AHEAD
  AHEAD=${AHEAD#* }
  if is_dropped "r50_p2_s$s"; then log "r50_p2 seed $s dropped (budget)"; continue; fi
  for k in $FOLDS; do
    if want "p2_f${k}_r50_s$s" r50_p2; then run "p2_f${k}_r50_s$s" resnet50 "data/p2_f$k" "$s" $R50_ARGS; fi
  done
done

budget_check slow_p2
if is_dropped slow_p2; then
  log "slow_p2 dropped (budget)"
else
  for k in $FOLDS; do
    if want "p2_f${k}_slow_s42" slow_p2; then run "p2_f${k}_slow_s42" slow "data/p2_f$k" 42 $SLOW_ARGS; fi
  done
fi

if [ "$DRY" = "1" ]; then
  n_r50=0; n_slow=0; n_probe=0; n_done=0; n_curve=0; dry_sec=0
  for k in $FOLDS; do
    for r in "p1_f${k}_r50_s42:r50_p1:r50" "p2_f${k}_r50_s42:r50_p2:r50" "p2_f${k}_r50_s43:r50_p2:r50" \
             "p2_f${k}_r50_s44:r50_p2:r50" "p2_f${k}_slow_s42:slow_p2:slow"; do
      IFS=: read -r name block rec <<< "$r"
      want "$name" "$block" || continue
      if [ "$block" = "slow_p2" ] && is_dropped slow_p2; then continue; fi
      if [ "$block" = "r50_p2" ] && is_dropped "r50_p2_s${name##*_s}"; then continue; fi
      if [ -f "$OUT/state/done_$name" ] && [ "$FORCE" != "1" ]; then n_done=$((n_done + 1)); continue; fi
      if [ "$rec" = "slow" ]; then n_slow=$((n_slow + 1)); else n_r50=$((n_r50 + 1)); fi
      c=$(curve_for "${name##*_s}"); n_curve=$((n_curve + c))
      if [ "$rec" = "slow" ]; then dry_sec=$((dry_sec + $(est_sec slow "$c"))); else dry_sec=$((dry_sec + $(est_sec resnet50 "$c"))); fi
    done
  done
  if want probe probe && { [ ! -f "$OUT/state/done_probe" ] || [ "$FORCE" = "1" ]; }; then n_probe=1; fi
  echo "DRY SUMMARY: to run: probe=$n_probe r50=$n_r50 slow=$n_slow (already done, skipped: $n_done); dropped by budget:${DROPPED:- none}"
  echo "DRY SUMMARY: EPOCH_CURVE=$EPOCH_CURVE: $n_curve of $((n_r50 + n_slow)) training runs get every epoch evaluated on test, the others best (val, test) + final epoch (test)"
  echo "DRY SUMMARY: projected wall time $(( n_probe * PROBE_EST_MIN + dry_sec / 60 )) min (estimates per run incl. evaluation: probe $PROBE_EST_MIN; r50 $R50_EST_MIN with curve, $R50_NOCURVE_EST_MIN without; slow $SLOW_EST_MIN with curve, $SLOW_NOCURVE_EST_MIN without)"
  if [ -n "$MISSING_ROOTS" ]; then
    echo "DRY SUMMARY: MISSING inputs:$MISSING_ROOTS"
    exit 2
  fi
  echo "DRY SUMMARY: all needed roots and inputs exist"
  exit 0
fi
log "incomplete:${INCOMPLETE:- none}  dropped:${DROPPED:- none}"
log "ALL_DONE total=$(( $(date +%s) - T_START ))s ($(elapsed_min) min)"
