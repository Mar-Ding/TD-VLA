#!/bin/bash
# P0 check J -- prove the GPU2 lock actually excludes, against the SAME lock file the
# existing driver holds.
#
# This is deliberately not "does flock exist". It is:
#   1. holder A takes the lock and sleeps;
#   2. while A holds it, a SECOND process tries `flock -n`;
#   3. the second attempt MUST FAIL (exit != 0);
#   4. after A releases, a third attempt MUST SUCCEED.
#
# Step 3 alone would also pass if the lock path were typo'd into a nonexistent directory
# for BOTH attempts, so step 4 is what proves the file is real and usable. And step 0
# asserts the path is the same file the other driver uses -- otherwise a "PASS" here
# would certify a lock nobody else takes.
set -u
LOCK=${GPU2_LOCK:-$HOME/sibling_pkg/audits/gpu2_rollout.lock}
OTHER=$HOME/sibling_pkg/scripts/rollout_locked_gpu2.sh
fail=0
chk(){ printf '  [%s] %s%s\n' "$1" "$2" "${3:-}"; [ "$1" = PASS ] || fail=1; }

echo "== J. GPU2 rollout lock mutual exclusion =="
echo "   lock path: $LOCK"

# --- 0. the lock we take is the lock the other driver takes -------------------------
if [ -f "$OTHER" ]; then
    their=$(grep -m1 '^LOCK=' "$OTHER" | cut -d= -f2-)
    # Their path is RELATIVE; resolve it the way their driver does (cwd = ~/sibling_pkg).
    resolved="$HOME/sibling_pkg/${their}"
    if [ "$resolved" = "$LOCK" ]; then
        chk PASS "the lock we take IS the file the existing sibling_pkg driver takes"
    else
        chk FAIL "lock mismatch: theirs resolves to '$resolved', ours is '$LOCK'" \
            " <- mutual exclusion would be a fiction"
    fi
else
    chk FAIL "other driver's lock script not found at $OTHER"
fi

mkdir -p "$(dirname "$LOCK")" || chk FAIL "cannot create lock dir"

# --- 1+2. holder A holds; a second attempt must be refused --------------------------
( exec 9>"$LOCK"; flock -x 9; echo held > /tmp/.gpu2_lock_held; sleep 5 ) &
A=$!
for _ in $(seq 1 50); do [ -f /tmp/.gpu2_lock_held ] && break; sleep 0.1; done
if [ ! -f /tmp/.gpu2_lock_held ]; then
    chk FAIL "holder A never acquired the lock (setup broken, test invalid)"
else
    if ( exec 8>"$LOCK"; flock -x -n 8 ) 2>/dev/null; then
        chk FAIL "a SECOND holder acquired the lock while A held it" \
            " <- GPU2 would run two rollouts at once"
    else
        chk PASS "second holder is REFUSED while the lock is held (flock -n fails)"
    fi
fi
wait $A 2>/dev/null
rm -f /tmp/.gpu2_lock_held

# --- 4. after release, acquisition must succeed -------------------------------------
if ( exec 7>"$LOCK"; flock -x -n 7 ) 2>/dev/null; then
    chk PASS "lock is acquirable again after release (the file is real and usable)"
else
    chk FAIL "lock still held after release -- stale lock or wrong path"
fi

echo
if [ "$fail" = 0 ]; then echo "  J RESULT: PASS"; else echo "  J RESULT: FAIL"; fi
exit $fail
