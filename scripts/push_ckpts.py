import os
from huggingface_hub import HfApi
token = open("/root/.hf_env").read().split("HF_TOKEN=")[1].strip()
api = HfApi(token=token)
repo = "AjelixAI/Ajelix-Fiber-130M"
tag = "tied-22L512d-b32-r64"
PLAN = {
    "stable-step40000-tokens21B": f"/root/phi/ckpt_{tag}_step40000.pt",
    "mid-anneal-step45000-tokens24B": f"/root/phi/ckpt_{tag}_step45000.pt",
    "mid-anneal-step50000-tokens26B": f"/root/phi/ckpt_{tag}_step50000.pt",
    "final-step52991-tokens27.8B": "/root/phi/ckpt_pretrain_" + tag + ".pt",
}
EARLY = {"periodic-checkpoints": [
    f"/root/phi/ckpt_{tag}_step{s}.pt" for s in
    (5000, 10000, 15000, 20000, 25000, 30000, 35000)]}
for br in list(PLAN) + list(EARLY):
    api.create_branch(repo_id=repo, branch=br, repo_type="model", exist_ok=True)
    print("branch:", br, flush=True)
for br, path in PLAN.items():
    api.upload_file(path_or_fileobj=path, path_in_repo="ckpt_" + tag + ".pt",
                    repo_id=repo, repo_type="model", revision=br)
    print("uploaded:", br, flush=True)
for br, paths in EARLY.items():
    for p in paths:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.path.basename(p), repo_id=repo,
                        repo_type="model", revision=br)
        print("uploaded:", br, os.path.basename(p), flush=True)
print("ALL-UPLOADED", flush=True)
