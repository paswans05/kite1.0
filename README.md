---
tags:
- compressed-tensors
license: other
license_name: modified-mit
library_name: transformers
pipeline_tag: image-text-to-text
---
<div align="center">
  <picture>
      <img src="figures/kite-logo.png" width="30%" alt="Kite 1.0">
  </picture>
</div>
<hr>
<div align="center" style="line-height:1">
  <a href="https://www.kite.ai" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-Kite%201.0-ff6b6b?color=1783ff&logoColor=white"/></a>
</div>

## 1. Model Introduction

Kite 1.0 is an open-source, native multimodal agentic model that advances practical capabilities in long-horizon coding, coding-driven design, proactive autonomous execution, and swarm-based task orchestration.

### Key Features
- **Long-Horizon Coding**: Kite 1.0 achieves significant improvements on complex, end-to-end coding tasks, generalizing robustly across programming languages (Rust, Go, Python) and domains spanning front-end, DevOps, and performance optimization.
- **Coding-Driven Design**: Kite 1.0 is capable of transforming simple prompts and visual inputs into production-ready interfaces and lightweight full-stack workflows, generating structured layouts, interactive elements, and rich animations with deliberate aesthetic precision.
- **Elevated Agent Swarm**: Scaling horizontally to 300 sub-agents executing 4,000 coordinated steps, Kite 1.0 can dynamically decompose tasks into parallel, domain-specialized subtasks, delivering end-to-end outputs from documents to websites to spreadsheets in a single autonomous run.
- **Proactive & Open Orchestration**: For autonomous tasks, Kite 1.0 demonstrates strong performance in powering persistent, 24/7 background agents that proactively manage schedules, execute code, and orchestrate cross-platform operations without human oversight.

## 2. Model Summary

<div align="center">

| | |
|:---:|:---:|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 1T |
| **Activated Parameters** | 32B |
| **Number of Layers** (Dense layer included) | 61 |
| **Number of Dense Layers** | 1 |
| **Attention Hidden Dimension** | 7168 |
| **MoE Hidden Dimension** (per Expert) | 2048 |
| **Number of Attention Heads** | 64 |
| **Number of Experts** | 384 |
| **Selected Experts per Token** | 8 |
| **Number of Shared Experts** | 1 |
| **Vocabulary Size** | 160K |
| **Context Length** | 256K |
| **Attention Mechanism** | MLA |
| **Activation Function** | SwiGLU |
| **Vision Encoder** | MoonViT |
| **Parameters of Vision Encoder** | 400M |
</div>

## 3. Evaluation Results

<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center"><sup>Kite 1.0</sup></th>
<th align="center"><sup>GPT-5.4 <br><sup>(xhigh)</sup></sup></th>
<th align="center"><sup>Claude Opus 4.6 <br><sup>(max effort)</sup></sup></th>
<th align="center"><sup>Gemini 3.1 Pro<br><sup>(thinking high)</sup></sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=5><strong>Agentic</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">HLE-Full<br>(w/ tools)</td>
<td align="center" style="vertical-align: middle">54.0</td>
<td align="center" style="vertical-align: middle">52.1</td>
<td align="center" style="vertical-align: middle">53.0</td>
<td align="center" style="vertical-align: middle">51.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">BrowseComp</td>
<td align="center" style="vertical-align: middle">83.2</td>
<td align="center" style="vertical-align: middle" rowspan="2">82.7</td>
<td align="center" style="vertical-align: middle" rowspan="2">83.7</td>
<td align="center" style="vertical-align: middle" rowspan="2">85.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">BrowseComp<br>(Agent Swarm)</td>
<td align="center" style="vertical-align: middle">86.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">DeepSearchQA<br>(f1-score)</td>
<td align="center" style="vertical-align: middle">92.5</td>
<td align="center" style="vertical-align: middle">78.6</td>
<td align="center" style="vertical-align: middle">91.3</td>
<td align="center" style="vertical-align: middle">81.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">DeepSearchQA<br>(accuracy)</td>
<td align="center" style="vertical-align: middle">83.0</td>
<td align="center" style="vertical-align: middle">63.7</td>
<td align="center" style="vertical-align: middle">80.6</td>
<td align="center" style="vertical-align: middle">60.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Toolathlon</td>
<td align="center" style="vertical-align: middle">50.0</td>
<td align="center" style="vertical-align: middle">54.6</td>
<td align="center" style="vertical-align: middle">47.2</td>
<td align="center" style="vertical-align: middle">48.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Claw Eval (pass^3)</td>
<td align="center" style="vertical-align: middle">62.3</td>
<td align="center" style="vertical-align: middle">60.3</td>
<td align="center" style="vertical-align: middle">70.4</td>
<td align="center" style="vertical-align: middle">57.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">OSWorld-Verified</td>
<td align="center" style="vertical-align: middle">73.1</td>
<td align="center" style="vertical-align: middle">75.0</td>
<td align="center" style="vertical-align: middle">72.7</td>
<td align="center" style="vertical-align: middle">-</td>
</tr>
<tr>
<td align="center" colspan=5><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Terminal-Bench 2.0</td>
<td align="center" style="vertical-align: middle">66.7</td>
<td align="center" style="vertical-align: middle">65.4*</td>
<td align="center" style="vertical-align: middle">65.4</td>
<td align="center" style="vertical-align: middle">68.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SWE-Bench Pro</td>
<td align="center" style="vertical-align: middle">58.6</td>
<td align="center" style="vertical-align: middle">57.7</td>
<td align="center" style="vertical-align: middle">53.4</td>
<td align="center" style="vertical-align: middle">54.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SWE-Bench Verified</td>
<td align="center" style="vertical-align: middle">80.2</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">80.8</td>
<td align="center" style="vertical-align: middle">80.6</td>
</tr>
</tbody>
</table>
</div>

## 4. Deployment

Currently, Kite 1.0 is recommended to run on the following inference engines:
* vLLM
* SGLang
* KTransformers

The version requirement for `transformers` is `>=4.57.1, <5.0.0`.

Deployment examples can be found in the [Model Deployment Guide](docs/deploy_guidance.md).

## 5. Model Usage

The usage demos below demonstrate how to call the API. 

### Chat Completion

This is a simple chat completion script which shows how to call the API in Thinking and Instant modes.

```python
import openai
import base64
import requests
def simple_chat(client: openai.OpenAI, model_name: str):
    messages = [
        {'role': 'system', 'content': 'You are Kite, an AI assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'which one is bigger, 9.11 or 9.9? think carefully.'}
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model_name, messages=messages, stream=False, max_tokens=4096
    )
    print('====== Below is reasoning content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')
```

## 6. Testing & Inference

You can run local tests on your fine-tuned or base model checkpoints using the provided Python scripts or import the text backbone into Ollama.

### 6.1 Multimodal Inference (With Image)
Test the model on an image using an online URL or local file path:
```bash
python test_kite.py \
    --model_path "./kite_qwen_trained" \
    --image_path "https://i.ibb.co/sdf0DN54/Nitro-Wallpaper-01-3840x2400.jpg" \
    --prompt "Describe what is in this image."
```

### 6.2 Text-Only Inference (Without Image)
To test conversational or text-only prompts, run the script without specifying an `--image_path`:
```bash
python test_kite.py \
    --model_path "./kite_qwen_trained" \
    --prompt "Who are you?"
```

Other predefined sample questions can be tested dynamically using `--prompt`.

### 6.3 Local Testing in Ollama (Text Backbone)
To run and test the conversational text backbone of Kite inside **Ollama**:

1. **Extract and Convert Configuration**:
   Run the utility script to generate a Qwen2-compatible text configuration and stage the model directory:
   ```bash
   python scratch/prepare_ollama_text.py
   ```
   This copies the weights, tokenizers, and creates a compatible `config.json` inside `./kite_qwen_text_only`.

2. **Create the Ollama Model**:
   Build the model using the generated `Modelfile`:
   ```bash
   ollama create kite1.0 -f Modelfile
   ```

3. **Run Inference**:
   Chat with your local Ollama model directly:
   ```bash
   ollama run kite1.0 "Who are you?"
   ```

## 7. License

Both the code repository and the model weights are released under the [Modified MIT License](LICENSE).

