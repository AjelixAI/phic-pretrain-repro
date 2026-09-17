#!/bin/bash
LOG=/root/phi/ckpt_backup.log
export SSHPASS='{6B?3@z^7adV+dNbNMKZ/&HaZ%W(3SyF'
BOXSSH="sshpass -e ssh -p 23 -o StrictHostKeyChecking=accept-new u546593-sub2@u546593.your-storagebox.de"
while true; do
  rsync -a --partial -e "sshpass -e ssh -p 23 -o StrictHostKeyChecking=accept-new" /root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_*.pt u546593-sub2@u546593.your-storagebox.de:sota1B/ >> $LOG 2>&1
  python3 - <<'PYEOF'
import glob, os, re
miles = sorted(glob.glob('/root/phi/ckpt_tied-16L2048d-b32-r64-sota1B_mile_*.pt'))
steps = [int(re.search(r'mile_(\d+)', m).group(1)) for m in miles]
landmarks = {0, 50000, 100000, 200000, 300000, 423900, 434079}
keep = set()
for i, s in enumerate(steps):
    if s % 7000 == 0 or s in landmarks or i >= len(steps) - 3:
        keep.add(miles[i])
for m in miles:
    if m not in keep:
        print(f"pruned {os.path.basename(m)}", flush=True)
        os.remove(m)
PYEOF
  sleep 600
done
