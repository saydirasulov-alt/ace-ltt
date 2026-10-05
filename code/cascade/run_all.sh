#!/usr/bin/env bash
# Cascade pipeline on the server, in protocol order (run from the folder that contains cascade/).
#
#   STAGE=selftest  bash cascade/run_all.sh   # statistics + every protocol guard (no GPU, ~3 min)
#   STAGE=ace-checklist  ACE-LTT pre-freeze checklist (declaration audit; reads no labels, changes nothing)
#   STAGE=ace-register   once: ACE dev study registration (ACE_DRY=1 verifies the preconditions only)
#   STAGE=ace-dev        once: the registered ACE dev run (refuses unless the registration still matches)
#   STAGE=ace2-register  once: ACE Study 2 (b ladder, spec v2.3) registration; ACE_DRY=1 verifies only
#   STAGE=ace2-dev       once: the registered Study 2 run (exploratory / method development)
#   STAGE=ace2-diag      paired post-registration diagnostic: edge-only vs the selected cascade.
#                        DESCRIPTIVE ONLY - no p-value, no decision rule, not in ANALYSIS_CODE
#   STAGE=ace-toy        ACE-LTT on a GENERIC synthetic instance: no dataset, no protocol, no detector.
#                        Method illustration for the paper; not in ANALYSIS_CODE, touches nothing registered
#   STAGE=protocol  bash cascade/run_all.sh   # 1. roles (calibration / sealed labels are NOT read) + protocol.yaml
#   STAGE=heads     bash cascade/run_all.sh   # 2. dev only: both YOLO26 heads; prints SELECTED by head_rule
#   (edit protocol.yaml: detector.nms = the selected head; other dev decisions)
#   STAGE=scores    bash cascade/run_all.sh   # 3. detector + VLM scores for all eval frames (label-free)
#   STAGE=records   bash cascade/run_all.sh   # 4. records for dev (sealed labels withheld)
#   STAGE=dev       bash cascade/run_all.sh   # 5. variant comparison on dev (last point where anything may change)
#   STAGE=freeze    bash cascade/run_all.sh   # 6. lock: protocol + splits + detector/agent CSVs + metas
#   STAGE=records   bash cascade/run_all.sh   # 7. rebuild records under the lock
#   STAGE=validity  bash cascade/run_all.sh   # 8. diagnostics inside calibration (refused before freeze)
#   STAGE=unseal    bash cascade/run_all.sh   # 9. ONCE: protocol receipt (atomic) -> unsealed records -> hash in receipt
#   STAGE=final     bash cascade/run_all.sh   # 10. evaluate once, only on the records hash in the receipt
#   after a crash in 9 or 10:  RESUME=1 STAGE=unseal|final ...   (same files; logged in the receipt)
#   STAGE=verify-final   read-only: final.md, final.json, unsealed records vs receipt; lock + code hash
#   STAGE=dataset_audit  label-free fig 2 input (WORK=<fsclean clean_work> adds the pHash curve; PC only)
#   RAVC-LTT (dev study, before freeze; MODEL/QUANT of the overlay + crop agent CSVs):
#   STAGE=crop-bytes     label-free: JPEG bytes of exactly the crop the VLM saw (CPU)
#   STAGE=ravc-register  once: protocol / code / input hashes + acceptance rule, with UTC time
#   STAGE=ravc-dev       once: dev-only RAVC report, refused unless the registration still matches
#   ESVA-LTT (dev study; MODEL/QUANT/VIEW of one agent CSV, default 3B overlay):
#   STAGE=diagnose       dev ceiling diagnostic + GO/NO-GO (run BEFORE registering ESVA)
#   STAGE=study-register once: closes a dev study's variant list + decision rule (cascade/study_register.py)
#   STAGE=probe          v2.9 dev probe: ranking (AUC), real lambda front, certifiability, temporal k-of-n
#                        AGENTS="overlay=<csv>,crop=<csv>,7b=<csv>" (existing score files only)
#   STAGE=esva-register  once: protocol esva block / code / input hashes + acceptance rule, with UTC time
#   STAGE=esva-dev       once: dev-only ESVA report vs edge_LTT / cascade_LTT
#   STAGE=figures_dev    figures from dev outputs only (+ head_report.json, dataset_audit.json)
#   STAGE=figures_final  figures from the hash-checked final.json only (also run automatically after final;
#                        a figure error never re-runs final). FIGFIX=1 -> --post-freeze-figure-fix
set -euo pipefail

N=${N:-~/PROJECT}
DATASET=${DATASET:-$N/lha-yolo26/datasets/FireSmoke-Clean_v0.2}
RUN=${RUN:-yolo26-base_s0}
WEIGHTS=${WEIGHTS:-$N/runs/fsc_v02/$RUN/weights/best.pt}
NMS=${NMS:-false}                 # must equal detector.nms in protocol.yaml
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
QUANT=${QUANT:-none}
VIEW=${VIEW:-overlay}
ADAPTIVE_VIEW=${ADAPTIVE_VIEW:-0} # dev-only RAVC experiment: score both overlay and crop, deploy one call
TRIALS=${TRIALS:-100}
DEVICE=${DEVICE:-0}
PURE_CLOUD=${PURE_CLOUD:-0}       # 1 = also run the detector-independent plain agent on every frame
                                  #     (protocol.yaml must then say pure_cloud_baseline: true)
STAGE=${STAGE:-selftest}
RESUME=${RESUME:-0}
FIGFIX=${FIGFIX:-0}
WORK=${WORK:-}
OUT=${OUT:-$N/runs/cascade}
PROTO=${PROTO:-$OUT/protocol}
AUDIT=${AUDIT:-$OUT/dataset_audit.json}
FIG_ARG=""
if [ "$FIGFIX" = 1 ]; then FIG_ARG="--post-freeze-figure-fix"; fi
RES_ARG=""
if [ "$RESUME" = 1 ]; then RES_ARG="--resume"; fi
DET_ARGS="--imgsz 640 --conf 0.001 --iou 0.7 --max-det 100"

export HF_HOME=${HF_HOME:-$N/hf_cache}          # keep model downloads inside $N (shared server)
AGENT_TAG="$(basename "$MODEL")-$QUANT-$VIEW"
DET=$OUT/det_${RUN}_nms-${NMS}.csv
AG=$OUT/agent_${AGENT_TAG}_${RUN}_nms-${NMS}.csv
AGO=$OUT/agent_$(basename "$MODEL")-${QUANT}-overlay_${RUN}_nms-${NMS}.csv
AGC=$OUT/agent_$(basename "$MODEL")-${QUANT}-crop_${RUN}_nms-${NMS}.csv
CB=$OUT/viewbytes_$(basename "$MODEL")-${QUANT}-crop_${RUN}_nms-${NMS}.csv
RAVC_OUT=$OUT/ravc_dev_${RUN}_nms-${NMS}_$(basename "$MODEL")-${QUANT}
RAVC_REG=$RAVC_OUT.registration.json
AGENTS=${AGENTS:-overlay=$AGO,crop=$AGC}
ESVA_OUT=$OUT/esva_dev_${RUN}_nms-${NMS}_${AGENT_TAG}
ESVA_REG=$ESVA_OUT.registration.json
REC=$OUT/records_${RUN}_nms-${NMS}_${AGENT_TAG}.csv
RES=$OUT/res_${RUN}_nms-${NMS}_${AGENT_TAG}
AGP=$OUT/agentplain_$(basename "$MODEL")-${QUANT}_${RUN}_nms-${NMS}.csv
PLAIN_ARGS=""
if [ "$PURE_CLOUD" = 1 ]; then PLAIN_ARGS="--agent-plain $AGP"; fi

case "$STAGE" in
  selftest)
    python -m cascade.selftest ;;
  ace-checklist|ace_checklist)
    python -m cascade.ace_ltt checklist --protocol "$OUT/protocol" ;;
  ace-register|ace_register)
    python -m cascade.ace_register --dataset "$DATASET" --protocol "$OUT/protocol" --det "$DET" \
      --agent crop="$AGENT_CROP" --agent overlay="$AGENT_OVERLAY" \
      --out "$OUT/ace.registration.json" ${ACE_DRY:+--dry-run} ;;
  ace-dev|ace_dev)
    python -m cascade.ace_dev --dataset "$DATASET" --protocol "$OUT/protocol" --det "$DET" \
      --agent crop="$AGENT_CROP" --agent overlay="$AGENT_OVERLAY" \
      --registration "$OUT/ace.registration.json" --out "$OUT/ace_dev_${TAG:-run}" ;;
  ace2-register|ace2_register)
    python -m cascade.ace2_register --dataset "$DATASET" --protocol "$OUT/protocol" --det "$DET" \
      --agent crop="$AGENT_CROP" --agent overlay="$AGENT_OVERLAY" \
      --out "$OUT/ace2.registration.json" ${ACE_DRY:+--dry-run} ;;
  ace2-dev|ace2_dev)
    python -m cascade.ace2_dev --dataset "$DATASET" --protocol "$OUT/protocol" --det "$DET" \
      --agent crop="$AGENT_CROP" --agent overlay="$AGENT_OVERLAY" \
      --registration "$OUT/ace2.registration.json" --out "$OUT/ace2_dev_${TAG:-run}" ;;
  ace-toy|ace_toy)
    python -m cascade.ace_toy --out "${OUT:-.}/ace_toy_${TAG:-run}" ;;
  ace2-diag|ace2_diag)
    python -m cascade.ace2_diag --dataset "$DATASET" --protocol "$OUT/protocol" --det "$DET" \
      --agent crop="$AGENT_CROP" --agent overlay="$AGENT_OVERLAY" \
      --registration "$OUT/ace2.registration.json" --run "$OUT/ace2_dev_${TAG:-run}" ;;
  protocol)
    mkdir -p "$OUT"
    python -m cascade.make_protocol init --dataset "$DATASET" --out "$PROTO" ;;
  heads)
    for h in false none; do
      python -m cascade.dump_detector --weights "$WEIGHTS" --dataset "$DATASET" --splits val --nms "$h" $DET_ARGS \
        --device "$DEVICE" --out "$OUT/det_dev_${RUN}_nms-$h.csv"
    done
    python -m cascade.head_report --dataset "$DATASET" --protocol "$PROTO" --weights "$WEIGHTS" --device "$DEVICE" \
      --det "$OUT/det_dev_${RUN}_nms-false.csv" "$OUT/det_dev_${RUN}_nms-none.csv" --json "$OUT/head_report.json" \
      | tee "$OUT/head_report.md" ;;
  scores)
    python -m cascade.dump_detector --weights "$WEIGHTS" --dataset "$DATASET" --out "$DET" --device "$DEVICE" \
      --nms "$NMS" $DET_ARGS
    python -m cascade.agent_vlm --dataset "$DATASET" --det "$DET" --out "$AG" --model "$MODEL" --quant "$QUANT" \
      --view "$VIEW" --device "cuda:$DEVICE"
    if [ "$ADAPTIVE_VIEW" = 1 ]; then
      python -m cascade.agent_vlm --dataset "$DATASET" --det "$DET" --out "$AGO" --model "$MODEL" --quant "$QUANT" \
        --view overlay --device "cuda:$DEVICE"
      python -m cascade.agent_vlm --dataset "$DATASET" --det "$DET" --out "$AGC" --model "$MODEL" --quant "$QUANT" \
        --view crop --device "cuda:$DEVICE"
    fi
    if [ "$PURE_CLOUD" = 1 ]; then
      python -m cascade.agent_vlm --dataset "$DATASET" --det "$DET" --out "$AGP" --model "$MODEL" --quant "$QUANT" \
        --view plain --s-min 0 --device "cuda:$DEVICE"
    fi
    echo "meta: ${DET%.csv}.meta.json  ${AG%.csv}.meta.json" ;;
  records)
    python -m cascade.build_records --dataset "$DATASET" --det "$DET" --agent "$AG" $PLAIN_ARGS --out "$REC" \
      --protocol "$PROTO" ;;
  dev)
    python -m cascade.run_cascade --records "$REC" --protocol "$PROTO" --mode dev --out-dir "$RES" --trials "$TRIALS" ;;
  diagnose)
    python -m cascade.dev_diagnose --dataset "$DATASET" --protocol "$PROTO" --det "$DET" --agent "$AG" \
      --out "$OUT/diagnose_${RUN}_nms-${NMS}_${AGENT_TAG}" --trials "${DIAG_TRIALS:-50}" ;;
  probe)
    python -m cascade.dev_probe --dataset "$DATASET" --protocol "$PROTO" --det "$DET" --agents "$AGENTS" \
      --out "$OUT/probe_${RUN}_nms-${NMS}${PROBE_TAG:-}" --trials "${PROBE_TRIALS:-20}" \
      --reference "${REFERENCE:-}" --boot "${BOOT:-1000}" ;;
  study_register|study-register)
    python -m cascade.study_register --name "${STUDY:-acquisition_ladder}" --protocol "$PROTO" --det "$DET" \
      --out "$OUT/${STUDY:-acquisition_ladder}.registration.json" --note "${NOTE:-}" ;;
  esva_register|esva-register)
    python -m cascade.esva_dev register --protocol "$PROTO" --det "$DET" --agent "$AG" --out "$ESVA_REG" ;;
  esva_dev|esva-dev)
    python -m cascade.esva_dev run --dataset "$DATASET" --protocol "$PROTO" --det "$DET" --agent "$AG" \
      --registration "$ESVA_REG" --out "$ESVA_OUT" --trials "$TRIALS" ;;
  crop_bytes|crop-bytes)
    python -m cascade.view_bytes --dataset "$DATASET" --det "$DET" --agent "$AGC" --out "$CB" ;;
  ravc_register|ravc-register)
    python -m cascade.ravc_dev register --protocol "$PROTO" --det "$DET" --overlay "$AGO" --crop "$AGC" \
      --crop-bytes "$CB" --out "$RAVC_REG" ;;
  ravc_dev|ravc-dev)
    python -m cascade.ravc_dev run --dataset "$DATASET" --protocol "$PROTO" --det "$DET" --overlay "$AGO" \
      --crop "$AGC" --crop-bytes "$CB" --registration "$RAVC_REG" --out "$RAVC_OUT" --trials "$TRIALS" ;;
  freeze)
    python -m cascade.make_protocol freeze --out "$PROTO" --det "$DET" --agent "$AG" $PLAIN_ARGS ;;
  validity)
    python -m cascade.run_cascade --records "$REC" --protocol "$PROTO" --mode validity --out-dir "$RES" \
      --trials "$TRIALS" ;;
  audit)
    python -m cascade.make_protocol audit --out "$PROTO" ;;
  codehash)
    python -m cascade.make_protocol code-hash ;;
  unseal)
    python -m cascade.build_records --dataset "$DATASET" --det "$DET" --agent "$AG" $PLAIN_ARGS \
      --out "${REC%.csv}_unsealed.csv" --protocol "$PROTO" --unseal $RES_ARG ;;
  final)
    python -m cascade.run_cascade --records "${REC%.csv}_unsealed.csv" --protocol "$PROTO" --mode final \
      --out-dir "$RES/final" $RES_ARG
    python -m cascade.figures final --protocol "$PROTO" \
      || echo "WARNING: figure step failed. final is done and is NOT re-run; fix and run STAGE=figures_final" ;;
  verify_final|verify-final)
    python -m cascade.make_protocol verify-final --out "$PROTO" ;;
  dataset_audit|dataset-audit)
    WA=""
    if [ -n "$WORK" ]; then WA="--work $WORK"; fi
    python -m cascade.dataset_audit --dataset "$DATASET" $WA --out "$AUDIT" ;;
  figures_dev|figures-dev)
    python -m cascade.figures dev --protocol "$PROTO" --dev-dir "$RES" --head-json "$OUT/head_report.json" \
      --dataset-audit "$AUDIT" --out "$RES/figures_dev" ;;
  figures_final|figures-final)
    python -m cascade.figures final --protocol "$PROTO" $FIG_ARG ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
echo "done: STAGE=$STAGE"
