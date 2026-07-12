import argparse
import os
from huggingface_hub import HfApi

HF_TOKEN   = "" # i have hardcoded my hf token as of now ....please dont fuck the repo
HF_REPO_ID = "godofwar1007/moelora"

def push(local_dir: str, repo_id: str, repo_folder: str, commit_message: str):
    if not os.path.exists(local_dir):
        print(f"ERROR: {local_dir} does not exist")
        return

    files = os.listdir(local_dir)
    print(f"Files in {local_dir}:")
    for f in files:
        size_mb = os.path.getsize(os.path.join(local_dir, f)) / 1e6
        print(f"  {f}  ({size_mb:.1f} MB)")

    print(f"\nPushing to {repo_id}/{repo_folder} ...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=repo_folder,   # this puts everything inside a subfolder
        commit_message=commit_message,
    )
    print(f"Done. View at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    required=True,          help="local dir to push")
    parser.add_argument("--folder", required=True,          help="folder name inside HF repo")
    parser.add_argument("--msg",    default="manual push",  help="commit message")
    parser.add_argument("--repo",   default=HF_REPO_ID,     help="HF repo id override")
    args = parser.parse_args()
    push(args.dir, args.repo, args.folder, args.msg)
