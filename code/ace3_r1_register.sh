#!/usr/bin/env bash
# Write the Study-3 R1 registration, once, with the gates in front of it.
#
#     cd ~/Norqobil/v249 && bash ace3_r1_register.sh
#
# The registration is written ONCE: `ace3_register` refuses an existing --out, and this script refuses first,
# so a second run cannot even start the module. Nothing here opens the certify fold: `ace3_register` loads the
# select uids only (see the two-stage loader), and the receipt it writes carries no audit metadata.
#
# Outside the `cascade` package and outside ANALYSIS_CODE: running it changes no digest.
#
# WHAT THIS SCRIPT PINS, and what it must not. It pins the identities that exist BEFORE it runs: the
# analysis-code digest, the protocol instance, and a clean lock. It deliberately pins NO registration digest -
# it is the script that CREATES the receipt, so any value pinned here could only be a stale one from another
# instance. The receipt's digest is established afterwards by `ace3_r1_receipt_audit.py`, and only then is it
# pinned into `ace3_prerun_check.py` and `ace3_run.sh`, which until that moment carry __UNSET_UNTIL_R1__ and
# refuse to run.

set -u
set -o pipefail

HOME_N="$HOME/Norqobil"
TREE="$HOME_N/v249"
PROTO="$HOME_N/runs/cascade/protocol_v249"
RUNS="$HOME_N/runs/cascade"
DS="$HOME_N/lha-yolo26/datasets/FireSmoke-Clean_v0.2"
DET="$RUNS/det_yolo26-base_s0_nms-none.csv"
AG_CROP="$RUNS/agent_Qwen2.5-VL-3B-Instruct-none-crop_yolo26-base_s0_nms-none.csv"
AG_OVER="$RUNS/agent_Qwen2.5-VL-3B-Instruct-none-overlay_yolo26-base_s0_nms-none.csv"
REG="$RUNS/ace3_v249.registration.json"
LOG="$TREE/r1_register_v249.log"
DRY="$TREE/r1_dryrun_v249.log"

EXPECT_CODE="aa461f6b8f1f2b556544e78680565ccc5cd4fbad56dbd8fe85185dd1069464b7"

say() { printf '%s\n' "$*"; }
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

say ""
say "=== this script ==="
say "  ace3_r1_register.sh  sha256  $(sha256sum "$0" | awk '{print $1}')"

say ""
say "=== gates, before anything is written ==="

[ -d "$TREE/cascade" ] || die "run this from ~/Norqobil/v249 (no cascade/ found at $TREE)"
cd "$TREE" || die "cannot cd to $TREE"

[ ! -e "$REG" ] || die "the registration already exists: $REG
       A registration is written once. If it must be replaced, that is a provenance decision,
       not a rerun: move the existing file aside deliberately and record why."
say "  registration absent            OK"

[ -e "$DRY" ] || die "the reviewed dry-run log is missing: $DRY
       The post-write audit compares the receipt against it, so it must be kept."
say "  dry-run log present            OK"

for f in "$PROTO/protocol.yaml" "$PROTO/splits.csv" "$PROTO/protocol.lock" \
         "$DS/manifest.csv" "$DET" "$AG_CROP" "$AG_OVER"; do
  [ -e "$f" ] || die "missing input: $f"
done
say "  all inputs present             OK"

[ ! -e "$PROTO/final_receipt.json" ] || die "an unseal/final receipt exists in $PROTO: the sealed fold has
       been touched and this study cannot be registered against this protocol instance."
say "  no unseal/final receipt        OK"

CODE=$(python3 -c "import sys;sys.path.insert(0,'.');from cascade.make_protocol import analysis_code;print(analysis_code()[0])") \
  || die "could not compute analysis_code_sha256"
[ "$CODE" = "$EXPECT_CODE" ] || die "analysis_code_sha256 is $CODE, expected $EXPECT_CODE"
say "  analysis_code                  OK  ${CODE:0:16}..."

LOCK=$(python3 -c "import sys;sys.path.insert(0,'.');from cascade.make_protocol import lock_status;print('|'.join(lock_status('$PROTO')) or 'CLEAN')") \
  || die "could not read the protocol lock"
[ "$LOCK" = "CLEAN" ] || die "lock_status is not clean: $LOCK"
say "  lock_status                    CLEAN"

say ""
say "=== writing the R1 registration ==="
say "  out : $REG"
say "  log : $LOG"

python3 -m cascade.ace3_register \
  --dataset "$DS" \
  --protocol "$PROTO" \
  --det "$DET" \
  --agent "crop=$AG_CROP" \
  --agent "overlay=$AG_OVER" \
  --out "$REG" \
  > "$LOG" 2>&1
RC=$?
say "  exit=$RC"
say ""
tail -8 "$LOG"

if [ "$RC" -ne 0 ]; then
  say ""
  die "the registration REFUSED (exit $RC). Nothing was registered; read $LOG.
       A design audit file may have been written next to the --out path - that is the guard working."
fi

[ -e "$REG" ] || die "exit 0 but no registration file at $REG"

say ""
say "=== the written receipt ==="
say "  sha256 $(sha256sum "$REG" | awk '{print $1}')"
say "  bytes  $(wc -c < "$REG")"

if [ -e "$TREE/ace3_r1_receipt_audit.py" ]; then
  say ""
  say "=== post-write audit ==="
  python3 "$TREE/ace3_r1_receipt_audit.py" "$REG" "$DRY"
  exit $?
fi

say ""
say "ace3_r1_receipt_audit.py is not next to this script; run the audit separately:"
say "  python3 ace3_r1_receipt_audit.py \"$REG\" \"$DRY\""
