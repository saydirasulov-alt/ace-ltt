#!/usr/bin/env bash
# Prepare protocol_v249 and predict Freeze 4. It does NOT freeze.
#
#     cd ~/Norqobil/v249 && bash ace3_v249_protocol_prepare.sh
#
# Five steps, in this order:
#   1. create ~/Norqobil/runs/cascade/protocol_v249, refusing if anything is already there;
#   2. copy EXACTLY TWO files from protocol_v248 with `cp -p` - protocol.yaml and splits.csv. No lock, no
#      receipt, no run artifact, no pre-freeze backup, nothing else;
#   3. prove the copies are byte-identical with `cmp`, against protocol_v248 AND against the other two
#      instances, and check both digests against the ones all three already carry;
#   4. assert protocol_v249 contains nothing but those two files, and that the three existing instances are
#      untouched;
#   5. predict Freeze 4 with `ace3_prefreeze_predict2.py`, naming protocol_v249 and aa461f6b… explicitly.
#
# WHAT IT DOES NOT DO. It does not run `freeze`, does not write into protocol/, protocol_v247/ or
# protocol_v248/, does not touch any registration or run artifact, and does not open any fold. Step 5 writes
# nothing at all. Freeze 4 stays a separate, explicit decision.

HOME_N="$HOME/Norqobil"
TREE="$HOME_N/v249"
RUNS="$HOME_N/runs/cascade"
SRC="$RUNS/protocol_v248"
DST="$RUNS/protocol_v249"
PREDICT="$TREE/ace3_prefreeze_predict2.py"

EXPECT_CODE="aa461f6b8f1f2b556544e78680565ccc5cd4fbad56dbd8fe85185dd1069464b7"
EXPECT_YAML="f585e490d9bc4aa047e8c91fc3e7f3d482560b7d8e21fedfa78f152e9cc02fa4"
EXPECT_SPLITS="362d8689b193d61b3f97ff08ea1acd1a0956824cbbea3544f96921940994d27f"

FAIL=0
ok()    { printf '  %-36s OK    %s\n' "$1" "${2-}"; }
no()    { printf '  %-36s FAIL  %s\n' "$1" "${2-}"; FAIL=$((FAIL + 1)); }
head_() { printf '\n=== %s ===\n' "$*"; }
sha()   { sha256sum "$1" | awk '{print $1}'; }

head_ "this script"
printf '  %-36s %s\n' "ace3_v249_protocol_prepare.sh" "$(sha "$0")"
printf '  %-36s %s\n' "date (UTC)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  %-36s %s\n' "source instance" "$SRC"
printf '  %-36s %s\n' "new instance" "$DST"

# ---------------------------------------------------------------- preconditions
head_ "preconditions"
[ -d "$TREE/cascade" ] || { no "run from $TREE" "no cascade/ there"; exit 1; }
cd "$TREE" || exit 1
CODE=$(python3 -c "import sys;sys.path.insert(0,'.');from cascade.make_protocol import analysis_code;print(analysis_code()[0])" 2>/dev/null)
[ "$CODE" = "$EXPECT_CODE" ] && ok "analysis_code" "${CODE:0:16}..." \
                             || { no "analysis_code" "is ${CODE:-unreadable}, expected $EXPECT_CODE"; exit 1; }
[ -d "$SRC" ] && ok "protocol_v248 present" || { no "protocol_v248" "missing"; exit 1; }
[ -e "$PREDICT" ] && ok "predict tool present" "$(sha "$PREDICT" | cut -c1-16)..." \
                  || no "predict tool" "$PREDICT is missing - step 5 cannot run"

if [ -e "$DST" ]; then
  no "protocol_v249 absent" "$DST already exists"
  printf '  It is not overwritten. If a previous attempt left it, move it aside deliberately:\n'
  printf '      mv %s %s.old.$(date -u +%%Y%%m%%dT%%H%%M%%SZ)\n' "$DST" "$DST"
  exit 1
fi
ok "protocol_v249 absent" "nothing to overwrite"

# ---------------------------------------------------------------- (1) and (2)
head_ "(1) create, (2) copy exactly two files with cp -p"
mkdir "$DST" || { no "mkdir $DST"; exit 1; }
for f in protocol.yaml splits.csv; do
  cp -p "$SRC/$f" "$DST/$f" || { no "cp -p $f"; exit 1; }
  ok "copied" "$f"
done
printf '\n  what is in %s now:\n' "$(basename "$DST")"
ls -la "$DST" | sed 's/^/    /'

# ---------------------------------------------------------------- (3)
head_ "(3) byte-identity, proved with cmp"
for f in protocol.yaml splits.csv; do
  cmp -s "$SRC/$f" "$DST/$f" && ok "cmp $f" "protocol_v248 == protocol_v249" \
                             || no "cmp $f" "the copy DIFFERS from protocol_v248"
done
for other in "$RUNS/protocol" "$RUNS/protocol_v247"; do
  if [ -d "$other" ]; then
    for f in protocol.yaml splits.csv; do
      cmp -s "$other/$f" "$DST/$f" && ok "cmp $f" "$(basename "$other") == protocol_v249" \
                                   || no "cmp $f" "$(basename "$other") != protocol_v249"
    done
  fi
done
GY=$(sha "$DST/protocol.yaml"); GS=$(sha "$DST/splits.csv")
printf '\n  protocol.yaml  %s\n  splits.csv     %s\n' "$GY" "$GS"
[ "$GY" = "$EXPECT_YAML" ]   && ok "protocol.yaml digest" "${GY:0:16}..." || no "protocol.yaml digest" "$GY"
[ "$GS" = "$EXPECT_SPLITS" ] && ok "splits.csv digest" "${GS:0:16}..."   || no "splits.csv digest" "$GS"

# ---------------------------------------------------------------- (4)
head_ "(4) nothing else came across, and the old instances are untouched"
EXTRA=$(cd "$DST" && ls -A | grep -vxE 'protocol\.yaml|splits\.csv' || true)
[ -z "$EXTRA" ] && ok "only the two files" "no lock, receipt, backup or run artifact" \
               || no "unexpected content in protocol_v249" "$EXTRA"
for f in protocol.lock receipt.json final_receipt.json protocol.yaml.prefreeze; do
  [ ! -e "$DST/$f" ] && ok "$f absent" || no "$f" "PRESENT in protocol_v249"
done
printf '\n  the three historical instances:\n'
for d in protocol protocol_v247 protocol_v248; do
  if [ -d "$RUNS/$d" ]; then
    printf '    %-16s lock=%s  receipt=%s  final=%s  yaml=%s\n' "$d" \
      "$([ -e "$RUNS/$d/protocol.lock" ] && echo present || echo absent)" \
      "$([ -e "$RUNS/$d/receipt.json" ] && echo present || echo absent)" \
      "$([ -e "$RUNS/$d/final_receipt.json" ] && echo PRESENT || echo absent)" \
      "$(sha "$RUNS/$d/protocol.yaml" | cut -c1-16)..."
  fi
done
[ -e "$RUNS/ace3_v249.registration.json" ] && no "v249 registration" "PRESENT" \
                                           || ok "v249 registration absent"
[ -d "$RUNS/ace3_v249_run" ] && no "v249 run dir" "PRESENT" || ok "v249 run dir absent"

# ---------------------------------------------------------------- (5)
head_ "(5) Freeze 4 prediction - writes nothing"
if [ -e "$PREDICT" ]; then
  python3 "$PREDICT" "$DST" "$EXPECT_CODE" > freeze4_prediction.log 2>&1
  RC=$?; RC_PRED=$RC
  cat freeze4_prediction.log
  [ "$RC" -eq 0 ] && ok "prediction" "freeze would SUCCEED" \
                  || no "prediction" "exit=$RC - freeze would REFUSE; read $TREE/freeze4_prediction.log"
  for need in "PROBLEMS        NONE" "WILL FILL       nothing" "code            $EXPECT_CODE"; do
    grep -qF "$need" freeze4_prediction.log && ok "predicted line" "$need" \
                                            || no "predicted line MISSING" "$need"
  done
  grep -qF "lock present    False | receipt False" freeze4_prediction.log \
    && ok "predicted line" "lock present False | receipt False" \
    || no "predicted line MISSING" "lock present False | receipt False"
else
  no "step 5 skipped" "the predict tool is not present"
fi

head_ "summary"
if [ "$FAIL" -eq 0 ]; then
  echo "  protocol_v249 PREPARED and Freeze 4 PREDICTED clean."
  echo "  protocol_v249 carries exactly protocol.yaml and splits.csv, byte-identical to all three earlier"
  echo "  instances. No lock, no receipt. A is CLOSED. NOTHING WAS FROZEN by this script."
  echo "  log: $TREE/freeze4_prediction.log"
  exit 0
fi
echo "  PREPARATION FAILED ($FAIL gate(s)). Do not freeze."
echo "  If protocol_v249 was created before the failure, inspect it and move it aside deliberately."
exit 1
