import yaml
import os
import torch
from nemo.collections.speechlm2.models import SALM

# --- 1. SETUP FUNCTIONS ---

def load_config(yaml_file="Transcription-Agent.yaml"):
    """Reads the configuration file."""
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file '{yaml_file}' not found.")
    
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config["agents"][0]

def load_model(model_name):
    """Loads the NVIDIA Canary model onto GPU or CPU."""
    print(f"Loading Model: {model_name}...")
    model = SALM.from_pretrained(model_name)
    model.eval() # Set to evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU (Expect slowness!)")
    
    return model

def transcribe(model, audio_path, max_tokens=2048):
    """Runs the transcription inference."""
    print(f"Transcribing file: {audio_path}")
    
    if not os.path.exists(audio_path):
        return f"Error: Audio file not found at {audio_path}"

    # Canary Prompt Structure
    prompt_content = f"Transcribe the following: {model.audio_locator_tag}"
    
    conversation = [
        {
            "role": "user",
            "content": prompt_content,
            "audio": [audio_path] 
        }
    ]

    # Generate
    try:
        output_ids = model.generate(
            prompts=[conversation],
            max_new_tokens=max_tokens
        )
        text = model.tokenizer.ids_to_text(output_ids[0].cpu())
        return text
    except Exception as e:
        return f"Error during inference: {e}"

# --- 2. MAIN EXECUTION ---

if __name__ == "__main__":
    try:
        # A. Load Settings
        agent_config = load_config()
        MODEL_NAME = agent_config["model_name"]
        AUDIO_FILE = agent_config["parameters"]["audio_file"]
        MAX_TOKENS = agent_config["parameters"].get("max_output_tokens", 1024)

        # B. Load Model (Only done once)
        canary_model = load_model(MODEL_NAME)

        # C. Run Transcription
        result = transcribe(canary_model, AUDIO_FILE, MAX_TOKENS)

        # D. Print Result
        print("\n" + "="*40)
        print("TRANSCRIPTION RESULT")
        print("="*40)
        print(result)
        print("="*40 + "\n")

    except Exception as e:
        print(f"\n Critical Error: {e}")