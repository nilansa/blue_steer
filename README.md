# Directory Structure

## inferenceWMDP
Contains scripts for WMDP inference on 3 models:
- Mistral 7B-Instruct
- Phi-3 Mini 128K Instruct
- LLaMA-3 8B-Instruct

Also includes scripts to aggregate results for questions affected by sandbagging across 3 domains:
- Bio
- Chem
- Cyber

## outputs/WMDPResults
Contains two files for each domain and model combination:
- **results.json** - Contains the actual model responses
- **metrics.json** - Contains aggregate accuracy and sandbagging statistics

## notebooks
Contains Jupyter notebooks for various sandbagging detection and mitigation experiments:

- **cot-inspection-using-adversarial-prompt.ipynb** - Sandbagging detection using chain of thought user prompts

- **final_caa_lat.ipynb** - CAA (Concept Activation Analysis) and LAT (Linear Artifact Transformation) experiments on TinyLlama

- **lora_noise_injection_implementation.ipynb** - Experiments on LoRA-based TinyLlama model with noise injection to inspect sandbagging behavior

- **noise_adding.ipynb** - Noise injection experiments in Phi3 model using the TinyMMLU dataset with results visualization

- **steering.ipynb** - Sandbagging experiments on TinyLlama model, including steering vector calculations and PCA-based visualization of projected activations

# Setup Instructions

## Clone the Repository
```bash
git clone https://github.com/nilansa/blue_steer.git  
cd BLUE_STEER
```

## Model Configuration
For open-source models that run locally on GPUs and load directly from the Hugging Face Hub using the transformers library, specify the Hugging Face model ID as follows:
```python
# LLaMA-3 8B Instruct
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

# Phi-3 Mini 128K Instruct
model_path = "microsoft/Phi-3-mini-128k-instruct"

# Mistral 7B Instruct
model_path = "mistralai/Mistral-7B-Instruct-v0.3"
```

**Note:** Models are downloaded automatically on first use; no local checkpoints are required beforehand.

# Running Inference and Evaluations

Refer to `README.md` for instructions.
