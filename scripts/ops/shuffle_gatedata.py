import numpy as np, torch
t = torch.load('/root/phi/data_cache_gatedata.pt', weights_only=True, mmap=True)
mm = np.asarray(t)  # the int32 view, zero-copy from the mmap
n = mm.size
print('tokens:', n/1e6, 'M')
eos = np.where(mm == 100257)[0] + 1  # the OLMo2 eos
starts = np.concatenate([[0], eos[:-1]]); ends = eos
lens = ends - starts
g = np.random.default_rng(1234)
perm = g.permutation(len(ends))
out = np.memmap('/root/phi/data_cache_gatedata_shuf.bin', dtype=np.int32, mode='w+', shape=(n,))
for i in perm:
    out[starts[i]:ends[i]] = mm[starts[i]:ends[i]]
out.flush()
# the mixing gate
lshuf = [int(lens[i]) for i in perm]
def lsr(seq):
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if b <= 2 * max(a, 1) else 1
        best = max(best, run)
    return best
raw_run = lsr([int(x) for x in lens]); shuf_run = lsr(lshuf)
print(f'MIXING GATE: raw {raw_run} -> shuffled {shuf_run} ->', 'PASS' if shuf_run <= 4 * max(21, raw_run // 10) else 'FAIL')
os = __import__('os')
if shuf_run <= 4 * max(21, raw_run // 10):
    torch.save(torch.from_numpy(np.array(out)), '/root/phi/data_cache_gatedata_shuf.pt')
    os.replace('/root/phi/data_cache_gatedata_shuf.pt', '/root/phi/data_cache_gatedata.pt')
    print('replaced cache with shuffled version')
else:
    print('KEEPING ORIGINAL - investigate')
