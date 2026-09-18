#!/bin/bash
# Box backup v3: copies via the CIFS mount (/mnt/hetzner_box). Temp-then-rename
# for atomic placement (soft-mount EIO can never produce a half-written ckpt
# under its final name). Milestones: immediate (checked every 60s). Running.pt:
# every 30 min. Throttled (nice/ionice) to stay off the training's I/O toes.
# If the mount is down: remount, retry next cycle - no SSH wrapper to hang.
LOG=/root/phi/ckpt_backup.log
D=/mnt/hetzner_box/sota1B
LOC=/root/phi
LAST_MILE=""
CYCLE=0
ensure_mount() {
  mountpoint -q /mnt/hetzner_box || { echo "$(date +%H:%M:%S) remounting" >> $LOG; mount /mnt/hetzner_box 2>>$LOG; }
}
cp_atomic() {  # src dst
  ionice -c2 -n7 nice -n10 cp "$1" "$D/.tmp_$(basename $2)" 2>>$LOG && mv "$D/.tmp_$(basename $2)" "$2"
}
while true; do
  ensure_mount
  if mountpoint -q /mnt/hetzner_box; then
    NEW_MILE=$(ls -t $LOC/ckpt_tied-16L2048d-b32-r64-sota1B_mile_*.pt 2>/dev/null | head -1)
    if [ -n "$NEW_MILE" ] && [ "$(basename $NEW_MILE)" != "$LAST_MILE" ]; then
      echo "$(date +%H:%M:%S) new milestone $(basename $NEW_MILE)" >> $LOG
      cp_atomic "$NEW_MILE" "$D/$(basename $NEW_MILE)" && LAST_MILE="$(basename $NEW_MILE)" && echo "$(date +%H:%M:%S) milestone synced" >> $LOG
    fi
    CYCLE=$((CYCLE+1))
    if [ $CYCLE -eq 1 ] || [ $((CYCLE % 30)) -eq 0 ]; then
      for f in $LOC/ckpt_tied-16L2048d-b32-r64-sota1B_running.pt $LOC/ckpt_tied-16L2048d-b32-r64-sota1B_mile_*.pt; do
        b=$(basename $f); [ -f "$D/$b" ] || { echo "$(date +%H:%M:%S) backfilling $b" >> $LOG; cp_atomic "$f" "$D/$b"; }
      done
      echo "$(date +%H:%M:%S) cycle $CYCLE: running.pt refresh" >> $LOG
      cp_atomic "$LOC/ckpt_tied-16L2048d-b32-r64-sota1B_running.pt" "$D/ckpt_tied-16L2048d-b32-r64-sota1B_running.pt"
    fi
  fi
  sleep 60
done
