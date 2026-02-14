import os
import torch
import json
from nemo.collections.speechlm2.models import SALM
# from tools.toolbox import tool 

# --- CONFIGURATION ---
MODEL_NAME = "nvidia/canary-qwen-2.5b"
_CACHED_MODEL = None

def _get_model():
    """
    Internal helper to load the NVIDIA Canary-Qwen-2.5B model only once.
    """
    global _CACHED_MODEL
    
    if _CACHED_MODEL is None:
        print(f"\n[System] Initializing Transcription Model: {MODEL_NAME}...")
        try:
            # Load the model
            model = SALM.from_pretrained(MODEL_NAME)
            model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                print(f"[System] Success! Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[System] WARNING: Running on CPU. This will be very slow.")
            
            _CACHED_MODEL = model
            
        except Exception as e:
            print(f"[System] FATAL ERROR loading model: {e}")
            raise e
    
    return _CACHED_MODEL

# @tool
def load_audio_file(file_path: str) -> dict:
    """
    Loads an audio file from the given file path, transcribes it using Canary-Qwen-2.5b,
    and returns the result as a dictionary.
    """
    # 1. Clean up the file path
    clean_path = file_path.strip().strip('"').strip("'")

    print(f"---- Loading audio file from {clean_path} ----")

    # 2. Check if file exists
    if not os.path.exists(clean_path):
        return {
            "status": "error",
            "message": f"File not found at: {clean_path}",
            "text": ""
        }

    try:
        # 3. Get the loaded model
        model = _get_model()

        # 4. Prepare the Prompt
        prompt_content = f"Transcribe the following: {model.audio_locator_tag}"
        
        conversation = [
            {
                "role": "user",
                "content": prompt_content,
                "audio": [clean_path] 
            }
        ]

        # 5. Run Inference
        print(f"[Transcriber] Transcribing {os.path.basename(clean_path)}...")
        
        output_ids = model.generate(
            prompts=[conversation],
            max_new_tokens=2048
        )
        
        # 6. Decode the result to text
        raw_text = model.tokenizer.ids_to_text(output_ids[0].cpu())

        # 7. Return Dictionary (Matches agent.py's expected input for json.dumps)
        return {
            "status": "success",
            "file_path": clean_path,
            "text": raw_text
        }

    except Exception as e:
        # 8. Return Error Dictionary
        return {
            "status": "critical_error",
            "message": str(e),
            "text": ""
        }




# --- STANDALONE TEST BLOCK ---
if __name__ == "__main__":
    # 1. Replace this with a real audio file path on your computer
    # Fixed: Added the 'r' prefix
    TEST_FILE = r"C:\Users\jimmy\Downloads\OSR_us_000_0060_8k.wav"
    
    print("\n" + "="*50)
    print("STARTING STANDALONE TRANSCRIPTION TEST")
    print("="*50)
    
    # Check if the test file exists first
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: You need to put a file named '{TEST_FILE}' in this folder to test!")
    else:
        # Run the tool directly
        # Note: If your decorator is correctly implemented with @functools.wraps,
        # you can call this just like a normal function.
        result = load_audio_file(TEST_FILE)
        
        print("\n--- TEST RESULT ---")
        print(json.dumps(result, indent=4))
        print("="*50)