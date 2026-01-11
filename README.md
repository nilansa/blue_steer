# Directory Structure

- InferenceWMDP - Contains scripts for WMDP inference on 3 models (Mistral 7B-Instruct, Phi-3 Mini 128K Instruct, and LLaMA-3 8B-Instruct) and scripts to aggregate results for questions affected by sandbagging across 3 domains (bio, chem, cyber).
- Outputs/WMDPResults - Contains two files each (results.json and metrics.json) for each domain and each model. The results.json file contains the actual model responses, while the metrics.json file contains aggregate accuracy and sandbagging statistics.

# Instructions

- Clone the repo:
    git clone https://github.com/nilansa/blue_steer.git  
    cd BLUE_STEER

- For open-source models that are run locally on GPUs and loaded directly from the Hugging Face Hub using the transformers library, specify the Hugging Face model ID as follows:

    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

    model_path = "microsoft/Phi-3-mini-128k-instruct"

    model_path = "mistralai/Mistral-7B-Instruct-v0.3"

  Models are downloaded automatically on first use; no local checkpoints are required beforehand.

# Inference/Evals

- Refer to experiments/README.md for instructions
