import json
import os

notebook_path = "train_kite_colab.ipynb"

# Load current notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Define new cells to insert
git_pull_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# If you already cloned, run this to get the latest updates and clean cache\n",
        "!git pull\n",
        "!rm -rf /root/.cache/huggingface/modules/transformers_modules/"
    ]
}

# Find the position of Step 1 clone code
clone_index = -1
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code" and any("git clone" in line for line in cell.get("source", [])):
        clone_index = i
        break

# We will remove any previous git pull cell first to avoid duplicates
git_pull_indices = [i for i, cell in enumerate(cells) if cell["cell_type"] == "code" and any("git pull" in line for line in cell.get("source", []))]
for idx in reversed(git_pull_indices):
    cells.pop(idx)

# Re-insert the git pull cell
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code" and any("git clone" in line for line in cell.get("source", [])):
        cells.insert(i + 1, git_pull_cell)
        print("Re-inserted git pull cell.")
        break

# Find Step 4 Fine-Tune markdown and training code
train_md_index = -1
train_code_index = -1
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown" and any("Step 4" in line for line in cell.get("source", [])):
        train_md_index = i
    if cell["cell_type"] == "code" and any("train_kite.py" in line for line in cell.get("source", [])):
        train_code_index = i

# Remove existing train cells under Step 4 to reset cleanly
step4_start = -1
for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown" and any("Step 4:" in line for line in cell.get("source", [])):
        step4_start = i
        break

if step4_start != -1:
    # Remove all following cells until Step 5
    idx = step4_start + 1
    while idx < len(cells):
        if cells[idx]["cell_type"] == "markdown" and any("Step 5:" in line for line in cells[idx].get("source", [])):
            break
        cells.pop(idx)
        
    # Re-insert training markdown and options
    step4_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Fine-Tune the Multimodal Projector (Training)\n",
            "Run `train_kite.py` in `projector_only` mode pointing to `./kite_qwen_base`. This freezes the pre-trained text backbone and trains only the Multimodal Projector and Vision Tower on the GPU.\n",
            "\n",
            "You can choose to train with a dummy dataset, a local JSON dataset, or load a dataset directly from the Hugging Face Hub."
        ]
    }
    
    option_a_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Option A: Train with the dummy dataset (quick run to verify pipeline)\n",
            "!python train_kite.py \\\n",
            "    --model_path \"./kite_qwen_base\" \\\n",
            "    --mode \"projector_only\" \\\n",
            "    --dummy \\\n",
            "    --epochs 1 \\\n",
            "    --lr 2e-5 \\\n",
            "    --batch_size 1 \\\n",
            "    --grad_accum 4 \\\n",
            "    --output_dir \"./kite_qwen_trained\""
        ]
    }
    
    option_b_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Option B: Train with actual local JSON dataset (change data_path to your uploaded dataset JSON)\n",
            "!python train_kite.py \\\n",
            "    --model_path \"./kite_qwen_base\" \\\n",
            "    --mode \"projector_only\" \\\n",
            "    --data_path \"./sample_data.json\" \\\n",
            "    --epochs 3 \\\n",
            "    --lr 2e-5 \\\n",
            "    --batch_size 2 \\\n",
            "    --grad_accum 4 \\\n",
            "    --output_dir \"./kite_qwen_trained\""
        ]
    }
    
    option_c_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Option C: Train loading a dataset directly from Hugging Face Hub (specify the repo and the target JSON file)\n",
            "# If you download COCO images, pass their folder path using --image_folder (otherwise, it falls back to dummy placeholders)\n",
            "# --max_samples limits the training data size to ensure it completes before Colab disconnects\n",
            "!pip install -q datasets\n",
            "!python train_kite.py \\\n",
            "    --model_path \"./kite_qwen_base\" \\\n",
            "    --mode \"projector_only\" \\\n",
            "    --data_path \"liuhaotian/LLaVA-Instruct-150K/llava_instruct_150k.json\" \\\n",
            "    --image_folder \"/content/train2017\" \\\n",
            "    --max_samples 15000 \\\n",
            "    --epochs 1 \\\n",
            "    --lr 2e-5 \\\n",
            "    --batch_size 2 \\\n",
            "    --grad_accum 4 \\\n",
            "    --output_dir \"./kite_qwen_trained\""
        ]
    }
    
    cells.insert(step4_start + 1, option_c_cell)
    cells.insert(step4_start + 1, option_b_cell)
    cells.insert(step4_start + 1, option_a_cell)
    print("Re-inserted Step 4 options (Dummy, Local JSON, and Hugging Face Dataset).")

# Clean existing Step 6 to reset cleanly
step6_indices = [i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown" and any("Step 6" in line for line in cell.get("source", []))]
if step6_indices:
    idx = step6_indices[0]
    while idx < len(cells):
        cells.pop(idx)

# Add Step 6 at the end
hf_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 6: Upload the Fine-Tuned Model to Hugging Face Hub\n",
        "Upload the folder of your fine-tuned model (`./kite_qwen_trained`) directly to your Hugging Face account with the model name `kite-i0.5b`."
    ]
}

hf_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Replace \"hf_YOUR_WRITE_TOKEN_HERE\" with your actual Hugging Face write token\n",
        "!python upload_to_hf.py \\\n",
        "    --folder_path \"./kite_qwen_trained\" \\\n",
        "    --repo_id \"paswans05/kite-i0.5b\" \\\n",
        "    --token \"hf_YOUR_WRITE_TOKEN_HERE\""
    ]
}

cells.append(hf_md_cell)
cells.append(hf_code_cell)
print("Appended Hugging Face upload cells.")

# Save notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook train_kite_colab.ipynb successfully updated!")
