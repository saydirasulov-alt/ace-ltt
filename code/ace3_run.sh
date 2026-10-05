#!/usr/bin/env bash
# THE ONE-SHOT EXECUTION. This is the command that opens the certify fold.
#
#     cd ~/Norqobil/v249 && bash ace3_run.sh
#
# Up to now nothing has read a certify label, score or feature. `ace3_dev` block (1) re-verifies the R1 receipt
# and re-derives the design from the select fold; block (2) calls load_certify and opens 𝒜; block (3) applies
# formal C3, PASS or REFUSE; block (4) executes one certify_family per arm at delta = 0.05.
#
# ORDER, as the code actually stands in v2.4.8: inside block (2)/(3) the audit receipt is COMPUTED first and
# WRITTEN last - `formal_c3` runs before `audit_path.write_text`, because the receipt embeds its verdict
# ("formal_C3": c3). So 𝒜 is opened, then the C3 verdict is computed, then one file carrying both is written.
# The receipt is therefore the first artifact on disk carrying a certify-fold quantity, but it is not written
# before C3 is decided. Nothing about the access boundary or the refuse-only invariant depends on that order;
# it is a provenance-statement question, recorded in §16/§17 of the change-set.
#
# It happens ONCE. `ace3_dev` refuses an existing run report and there is no resume path. This script refuses
# first, so a second attempt cannot even start the module. The log counts as an artifact of the run for that
# purpose: a first attempt that crashes BEFORE the audit receipt is written leaves no JSON behind, so the JSON
# gate alone would let a second attempt start. The log is created before python starts, with noclobber, so the
# reservation is atomic and its existence is the marker.
#
# Outside the `cascade` package and outside ANALYSIS_CODE: running it changes no digest.

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
OUT="$RUNS/ace3_v249_run"                     # the one out-dir; every artifact of the run lands here
LOG="$TREE/r1_execute_v249.log"

EXPECT_CODE="aa461f6b8f1f2b556544e78680565ccc5cd4fbad56dbd8fe85185dd1069464b7"
EXPECT_REG="d9bcf680f1374d8124fe7549147538af3722683feda6e72f8821dee8da0c6b42"

say() { printf '%s\n' "$*"; }
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

say ""
say "=== this script ==="
say "  ace3_run.sh  sha256  $(sha256sum "$0" | awk '{print $1}')"
say "  out-dir             $OUT"

say ""
say "=== gates, before the certify fold is opened ==="

[ -d "$TREE/cascade" ] || die "run this from ~/Norqobil/v249"
cd "$TREE" || die "cannot cd to $TREE"

for f in "$OUT/ace3_dev.json" "$OUT/ace3_audit_receipt.json" "$OUT/ace3_dev.md" "$OUT/ace3_REFUSED.json" \
         "$LOG"; do
  [ ! -e "$f" ] || die "$f exists: this study is executed ONCE. A second execution is not a rerun, even if
       the first attempt left no JSON behind; if the run must be repeated, that is a provenance decision to
       record deliberately - move the existing artifacts aside by hand and write down why."
done
say "  no run artifact yet            OK  (json, md and the log)"

# The registration digest cannot be pinned until the new R1 receipt exists and has been audited, so it ships
# UNSET and unset is a hard STOP - never a skipped check. Leaving a previous instance's digest here would be
# worse than leaving it blank: it would look pinned while verifying the wrong receipt. This sits after the
# one-shot gate so that an already-executed study reports THAT, which is the graver fact.
[ "$EXPECT_REG" != "__UNSET_UNTIL_R1__" ] || die "EXPECT_REG is still __UNSET_UNTIL_R1__.
       The v249 R1 receipt has not been written and audited yet, so there is no digest to verify against.
       Order: Freeze 4 -> ace3_register --dry-run -> design identity comparison -> real R1 -> R1 audit ->
       pin that audited sha256 into EXPECT_REG here and in ace3_prerun_check.py -> pre-run check -> this."
say "  EXPECT_REG pinned              OK  ${EXPECT_REG:0:16}..."

[ -e "$REG" ] || die "no registration at $REG"
GOT=$(sha256sum "$REG" | awk '{print $1}')
[ "$GOT" = "$EXPECT_REG" ] || die "the registration digest is $GOT, expected $EXPECT_REG"
say "  registration                  OK  ${GOT:0:16}..."

for f in "$PROTO/protocol.yaml" "$PROTO/splits.csv" "$PROTO/protocol.lock" \
         "$DS/manifest.csv" "$DET" "$AG_CROP" "$AG_OVER"; do
  [ -e "$f" ] || die "missing input: $f"
done
say "  all inputs present            OK"

[ ! -e "$PROTO/final_receipt.json" ] || die "an unseal/final receipt exists in $PROTO"
say "  no unseal/final receipt       OK"

CODE=$(python3 -c "import sys;sys.path.insert(0,'.');from cascade.make_protocol import analysis_code;print(analysis_code()[0])") \
  || die "could not compute analysis_code_sha256"
[ "$CODE" = "$EXPECT_CODE" ] || die "analysis_code_sha256 is $CODE, expected $EXPECT_CODE"
say "  analysis_code                 OK  ${CODE:0:16}..."

LOCK=$(python3 -c "import sys;sys.path.insert(0,'.');from cascade.make_protocol import lock_status;print('|'.join(lock_status('$PROTO')) or 'CLEAN')") \
  || die "could not read the protocol lock"
[ "$LOCK" = "CLEAN" ] || die "lock_status is not clean: $LOCK"
say "  lock_status                   CLEAN"

say ""
say "=== reserving the log, atomically, BEFORE python starts ==="
mkdir -p "$(dirname "$LOG")" || die "cannot create $(dirname "$LOG")"
( set -o noclobber; : > "$LOG" ) 2>/dev/null \
  || die "could not reserve the log $LOG - it already exists.
       The reservation is the one-shot marker and it is taken before the module is started, so this means an
       execution has already begun. It is not a rerun: move the existing artifacts aside deliberately."
say "  log reserved                  OK  $LOG"

say ""
say "=== EXECUTING. The certify fold is opened from here. ==="
say ""

python3 -m cascade.ace3_dev \
  --dataset "$DS" \
  --protocol "$PROTO" \
  --det "$DET" \
  --agent "crop=$AG_CROP" \
  --agent "overlay=$AG_OVER" \
  --registration "$REG" \
  --out "$OUT" \
  >> "$LOG" 2>&1
RC=$?
say "  exit=$RC"
say ""
tail -25 "$LOG"

say ""
say "=== artifacts ==="
for f in ace3_audit_receipt.json ace3_dev.json ace3_dev.md ace3_REFUSED.json; do
  if [ -e "$OUT/$f" ]; then
    say "  $f  $(sha256sum "$OUT/$f" | awk '{print $1}')  $(wc -c < "$OUT/$f") bytes"
  else
    say "  $f  absent"
  fi
done

if [ "$RC" -ne 0 ]; then
  say ""
  say "the run did not complete (exit $RC). If ace3_audit_receipt.json exists, 𝒜 was recorded and formal C3"
  say "refused - a fail-closed outcome. If it does not, the run stopped before 𝒜 was opened. Read $LOG."
fi

if [ -e "$TREE/ace3_run_audit.py" ]; then
  say ""
  say "=== post-run audit ==="
  python3 "$TREE/ace3_run_audit.py" "$OUT" "$REG"
  exit $?
fi

say ""
say "ace3_run_audit.py is not next to this script; run the audit separately:"
say "  python3 ace3_run_audit.py \"$OUT\" \"$REG\""
exit "$RC"
