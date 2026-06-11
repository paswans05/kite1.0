# coding=utf-8
import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("[ERROR] huggingface_hub package is not installed. Please install it by running:")
    print("pip install huggingface_hub")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Upload a trained model or configuration folder to the Hugging Face Hub.")
    parser.add_argument("--folder_path", type=str, default="./kite_qwen_trained",
                        help="Path to the local folder containing model weights and configs (default: ./kite_qwen_trained)")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="The target repository ID on Hugging Face (e.g., 'your_username/kite-qwen-0.5b')")
    parser.add_argument("--token", type=str, default=None,
                        help="Your Hugging Face write token (if not logged in via CLI)")
    parser.add_argument("--private", action="store_true",
                        help="Create the repository as private (default: public)")
    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        print(f"[ERROR] Local folder '{args.folder_path}' does not exist.")
        sys.exit(1)

    print(f"Connecting to Hugging Face Hub...")
    api = HfApi(token=args.token)

    try:
        # Step 1: Create repository on HF if it does not exist
        print(f"Ensuring repository '{args.repo_id}' exists on HF Hub...")
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            repo_type="model",
            exist_ok=True
        )
        print(f"[SUCCESS] Repository '{args.repo_id}' is ready.")
    except Exception as e:
        print(f"[ERROR] Failed to ensure repository existence: {e}")
        print("Please check your Hugging Face token permissions (it must have WRITE access).")
        sys.exit(1)

    try:
        # Step 2: Upload all files in the folder
        print(f"Uploading files from '{args.folder_path}' to HF Hub...")
        api.upload_folder(
            folder_path=args.folder_path,
            repo_id=args.repo_id,
            repo_type="model",
            token=args.token
        )
        print("=" * 60)
        print("[SUCCESS] All files successfully uploaded!")
        print(f"View your model at: https://huggingface.co/{args.repo_id}")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Failed to upload files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
