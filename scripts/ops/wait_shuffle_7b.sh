#!/bin/bash
while pgrep -f sample_dolmino.py > /dev/null; do sleep 20; done
tail -2 /root/phi/sample_7b.log
/root/venv/bin/python /root/phi/shuffle_dolmino.py /root/phi/data_cache_dolmino_7B_raw.pt /root/phi/data_cache_dolmino_7B_shuf.pt
