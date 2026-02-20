import torch
import os
from nemo.collections.speechlm2.models import SALM

# Load the model once during initialization to keep it in VRAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SALM.from_pretrained('nvidia/canary-qwen-2.5b').bfloat16().eval().to(device)

def load_audio_file(file_path: str) -> str:
    """
    Transcribes audio using the locally hosted Canary-Qwen-2.5B model.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        # Canary-Qwen uses a specific prompt format for ASR mode
        prompt_text = f"Transcribe the following: {model.audio_locator_tag}"
        
        # Generate transcription
        answer_ids = model.generate(
            prompts=[[{"role": "user", "content": prompt_text, "audio": [file_path]}]],
            max_new_tokens=2048
        )
        
        # Decode and return the result
        return model.tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"Transcription error with Canary-Qwen: {str(e)}"