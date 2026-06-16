import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert or restore config.json inside ./kite1.0 for Ollama compatibility.")
    parser.add_argument("--restore", action="store_true", help="Restore config.json to VLM configuration from config_vlm.json")
    args = parser.parse_args()

    model_dir = "./kite1.0"
    config_path = os.path.join(model_dir, "config.json")
    backup_path = os.path.join(model_dir, "config_vlm.json")

    if args.restore:
        if not os.path.exists(backup_path):
            print(f"Error: Backup config {backup_path} does not exist. Cannot restore.")
            return
        # Restore
        if os.path.exists(config_path):
            os.remove(config_path)
        os.rename(backup_path, config_path)
        print("Successfully restored VLM configuration inside ./kite1.0")
        return

    # Otherwise, perform the conversion
    if not os.path.exists(config_path):
        print(f"Error: Model config {config_path} does not exist.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # If it is already a Qwen2 model and VLM config is not backed up yet, warning
    if config.get("model_type") == "qwen2":
        print("Warning: config.json is already formatted as a Qwen2 model. Check if backup is already present.")
    else:
        # Save backup
        with open(backup_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Backed up VLM configuration to {backup_path}")

    # Extract text model configuration
    text_config = config.get("text_config", {})
    if not text_config and config.get("model_type") == "kite":
        print("Error: Could not find 'text_config' in config.json")
        return
    elif not text_config:
        # If it was already converted, maybe the whole config is the text config
        text_config = config

    # Override/add necessary parameters for Qwen2 compatibility in Ollama
    text_config["architectures"] = ["Qwen2ForCausalLM"]
    text_config["model_type"] = "qwen2"
    text_config["torch_dtype"] = "bfloat16"

    # Write the modified config back to config.json
    with open(config_path, "w") as f:
        json.dump(text_config, f, indent=2)

    print("Created config.json for Ollama Qwen2 compatibility inside ./kite1.0")

    # Update Modelfile FROM instruction
    modelfile_content = """# Ollama Modelfile for Kite 1.0 (Text Backbone Only)
FROM ./kite1.0

# Set inference parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_user|>"
PARAMETER stop "<|im_assistant|>"

# Set the system instruction
SYSTEM "You are Kite, a helpful assistant."
"""

    with open("Modelfile", "w") as f:
        f.write(modelfile_content)

    print("Ollama Modelfile successfully updated.")

if __name__ == "__main__":
    main()
