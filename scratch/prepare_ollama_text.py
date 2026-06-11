import os
import json
import shutil

src_dir = "./kite_qwen_trained"
dst_dir = "./kite_qwen_text_only"

if not os.path.exists(src_dir):
    print(f"Error: Source directory {src_dir} does not exist.")
    exit(1)

os.makedirs(dst_dir, exist_ok=True)

# Copy the weights and tokenizers
files_to_copy = [
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json"
]

for file_name in files_to_copy:
    src = os.path.join(src_dir, file_name)
    dst = os.path.join(dst_dir, file_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {file_name}")

# Read and modify config.json
with open(os.path.join(src_dir, "config.json"), "r") as f:
    config = json.load(f)

# Extract only the text model configuration parameters
text_config = config.get("text_config", {})

# Override/add necessary parameters for Qwen2
text_config["architectures"] = ["Qwen2ForCausalLM"]
text_config["model_type"] = "qwen2"
text_config["torch_dtype"] = "bfloat16"

with open(os.path.join(dst_dir, "config.json"), "w") as f:
    json.dump(text_config, f, indent=2)

print("Created config.json for Ollama Qwen2 compatibility.")

# Create the Modelfile for text-only model
modelfile_content = """# Ollama Modelfile for Kite 1.0 (Text Backbone Only)
FROM ./kite_qwen_text_only

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
