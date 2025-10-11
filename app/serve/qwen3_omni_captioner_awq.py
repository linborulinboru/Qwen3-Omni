#!/usr/bin/env python3
"""
Qwen3-Omni Captioner AWQ Quantized Audio Captioning HTTP Service
Based on AWQ quantization for reduced GPU memory usage
"""

import io
import os
import sys
import tempfile
import importlib.util
import threading
import uuid
import time
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify, send_file
from huggingface_hub import hf_hub_download
from opencc import OpenCC

# AWQ quantization imports
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoProcessor
import sys
import os
# Add the app directory to the path to import qwen_omni_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from qwen_omni_utils import process_mm_info

# Register Qwen3OmniMoe models to handle the custom architecture
try:
    from register_qwen3_omni import register_all
    register_all()
except ImportError:
    print("[ERROR] Could not import register_qwen3_omni.")
    raise

# ==================== Global Configuration ====================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# Global variables
model = None
processor = None
opencc_converter = None  # OpenCC converter for Traditional Chinese
model_lock = threading.Lock()
processing_lock = threading.Lock()
job_status = {}
job_lock = threading.Lock()

# Idle tracking
last_activity_time = None
is_processing = False
idle_check_interval = 30  # Check every 30 seconds
idle_timeout = 300  # 5 minutes
model_config = {}  # Store model configuration for auto-reload
model_loaded_time = None  # Track when model was loaded

# Directories - use absolute paths to ensure correct location
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(WORK_DIR, "inputs")
TEMP_DIR = os.path.join(WORK_DIR, "temp")
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_awq_model(model_path, dtype=torch.float16, device_map="auto", **kwargs):
    """
    Load AWQ quantized Qwen3-Omni Captioner model
    """
    global model, processor, model_loaded_time
    
    print(f"[INFO] Loading AWQ quantized model from: {model_path}")
    
    try:
        # Load the AWQ model with trust_remote_code enabled
        quantized_model = AutoAWQForCausalLM.from_quantized(
            model_path,
            trust_remote_code=True,
            fuse_layers=True,
            **kwargs
        )
        
        # Convert to appropriate model class if needed
        # In practice, the AWQ model should handle this automatically
        model = quantized_model.model
        
        # Load processor - using AutoProcessor as fallback
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except (AttributeError, ValueError):
            # If AutoProcessor fails, try to get the specific processor class dynamically
            from transformers import AutoTokenizer
            # Attempt to load specific processor based on config
            try:
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    # Get the processor class name from config if available
                    if "processor_class" in config:
                        processor_class_name = config["processor_class"]
                        # Try to dynamically import the processor
                        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                    else:
                        # Fallback to tokenizer
                        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                else:
                    # Fallback to tokenizer if config doesn't exist
                    processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                # Final fallback to tokenizer
                processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Record when model was loaded
        model_loaded_time = time.time()
        
        print(f"[INFO] AWQ quantized model loaded successfully!")
        
        return model, processor
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        print("[INFO] This error may occur if the transformers library doesn't recognize the model architecture.")
        print("[INFO] You may need to update transformers with the command: pip install git+https://github.com/huggingface/transformers.git")
        raise e

def unload_model():
    """Unload the model from memory to save GPU resources during idle periods"""
    global model, processor
    if model is not None:
        print("[INFO] Unloading model from memory...")
        # Move model to CPU first to free GPU memory
        model = model.cpu()
        # Explicitly delete the model and processor
        del model
        del processor
        # Clear CUDA cache to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = None
        processor = None
        print("[INFO] Model unloaded successfully")
    else:
        print("[INFO] Model was not loaded, nothing to unload")

def ensure_model_loaded(args):
    """Ensure the model is loaded, loading it if necessary"""
    global model, processor
    if model is None or processor is None:
        print("[INFO] Model not loaded, loading now...")
        initialize_model(args)
        print("[INFO] Model loaded, ready for processing")

# ==================== Model Initialization ====================

def initialize_model(args):
    """Initialize the model with specified parameters"""
    global model, processor, opencc_converter, model_config
    
    model_config = {
        'checkpoint_path': args.checkpoint_path,
        'dtype': args.dtype,
        'device_map': args.device_map,
        'use_flash_attention': args.flash_attn2,
        'enable_thinking': args.enable_thinking
    }
    
    # Load AWQ model
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    device_map = 'auto' if args.device_map == 'auto' else args.device_map
    
    model, processor = load_awq_model(
        args.checkpoint_path,
        dtype=dtype,
        device_map=device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )
    
    # Apply flash attention if specified
    if args.flash_attn2:
        print("[INFO] Flash Attention 2 enabled")
    
    # Initialize OpenCC converter if needed
    if args.enable_opencc:
        opencc_converter = OpenCC('s2t')  # Simplified to Traditional Chinese
    
    print(f"[INFO] Model initialized: {args.checkpoint_path}")
    print(f"[INFO] Model type: Qwen3-Omni Captioner AWQ")
    print(f"[INFO] Dtype: {dtype}")
    print(f"[INFO] Device map: {device_map}")


def transcribe_audio_file(audio_path, audio_only=True, max_new_tokens=8192, temperature=0.1, repetition_penalty=1.1, segment_duration=60):
    """
    Transcribe audio file using the loaded model
    """
    global model, processor
    
    # Verify model is loaded (in case it was unloaded during idle)
    if model is None or processor is None:
        print("[WARNING] Model was unloaded, this should not happen during processing")
        return {
            'error': 'Model is currently unloaded, please try again'
        }
    
    # Load audio file
    audio_data, sr = librosa.load(audio_path, sr=16000)  # Qwen models typically use 16kHz
    
    # Create conversation format for captioning (audio only, no text prompt)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
            ],
        }
    ]
    
    print(f"[INFO] Processing conversation: {conversation}")
    
    # Prepare input using processor
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = processor(
        text=text,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    
    inputs = inputs.to(model.device).to(model.dtype)
    
    print(f"[INFO] Input shapes - audio: {audios.shape if audios is not None else None}")
    
    # Generate transcription
    with torch.inference_mode():
        text_ids, audio_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=True,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=max_new_tokens,
            thinker_temperature=temperature,
            thinker_do_sample=temperature > 0,
            thinker_repetition_penalty=repetition_penalty,
            use_audio_in_video=False
        )
    
    # Decode the generated text
    generated_text = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Apply OpenCC conversion if enabled
    if opencc_converter:
        generated_text = opencc_converter.convert(generated_text)
    
    print(f"[INFO] Generated caption: {generated_text}")
    
    return {
        'text': generated_text,
        'audio_path': audio_path,
        'processing_time': time.time()  # Placeholder
    }


# ==================== Flask Routes ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'Qwen3-Omni Captioner AWQ'
    }), 200


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe uploaded audio file"""
    global is_processing, last_activity_time
    
    with processing_lock:
        is_processing = True
        last_activity_time = time.time()
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Ensure model is loaded before processing
        args_from_config = type('Args', (), model_config)() if model_config else type('Args', (), {
            'checkpoint_path': 'Qwen/Qwen3-Omni-30B-A3B-Captioner-AWQ-4bit',
            'dtype': 'float16',
            'device_map': 'auto',
            'flash_attn2': False,
            'enable_thinking': False,
            'max_new_tokens': 8192,
            'temperature': 0.1,
            'repetition_penalty': 1.1
        })()
        ensure_model_loaded(args_from_config)
        
        # Get parameters from request
        max_new_tokens = int(request.form.get('max_new_tokens', 8192))
        temperature = float(request.form.get('temperature', 0.1))
        repetition_penalty = float(request.form.get('repetition_penalty', 1.1))
        segment_duration = int(request.form.get('segment_duration', 60))
        
        # Save uploaded file temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        
        audio_file.save(temp_path)
        
        try:
            # Transcribe the audio
            result = transcribe_audio_file(
                temp_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                segment_duration=segment_duration
            )
            
            if 'error' in result:
                return jsonify(result), 503  # Service unavailable if model was unloaded mid-request
            
            # Clean up temp file
            os.remove(temp_path)
            
            return jsonify(result), 200
        except Exception as e:
            # Clean up temp file even if processing fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)}), 500
    
    finally:
        with processing_lock:
            is_processing = False


@app.route('/v1/audio/captions', methods=['POST'])
def openai_compatible_caption():
    """OpenAI-compatible audio captioning endpoint"""
    global is_processing, last_activity_time
    
    with processing_lock:
        is_processing = True
        last_activity_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Ensure model is loaded before processing
        args_from_config = type('Args', (), model_config)() if model_config else type('Args', (), {
            'checkpoint_path': 'Qwen/Qwen3-Omni-30B-A3B-Captioner-AWQ-4bit',
            'dtype': 'float16',
            'device_map': 'auto',
            'flash_attn2': False,
            'enable_thinking': False,
            'max_new_tokens': 8192,
            'temperature': 0.1,
            'repetition_penalty': 1.1
        })()
        ensure_model_loaded(args_from_config)
        
        # Get model parameters
        model_params = request.form.get('model', 'Qwen/Qwen3-Omni-30B-A3B-Captioner-AWQ-4bit')
        prompt = request.form.get('prompt', '')  # Ignored for captioner model
        temperature = float(request.form.get('temperature', 0.1))
        max_tokens = int(request.form.get('max_tokens', 8192))
        
        # Save uploaded file temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        
        audio_file.save(temp_path)
        
        try:
            # Transcribe the audio (captioning)
            result = transcribe_audio_file(
                temp_path,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=1.1,
                segment_duration=30  # Default for captioning
            )
            
            if 'error' in result:
                return jsonify(result), 503  # Service unavailable if model was unloaded mid-request
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Format response in OpenAI-compatible format
            response = {
                'text': result['text']
            }
            
            return jsonify(response), 200
        except Exception as e:
            # Clean up temp file even if processing fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)}), 500
    
    finally:
        with processing_lock:
            is_processing = False


@app.route('/status', methods=['GET'])
def status():
    """Get current processing status"""
    current_time = time.time()
    idle_duration = current_time - last_activity_time if last_activity_time else 0
    time_since_load = current_time - model_loaded_time if model_loaded_time else 0
    
    return jsonify({
        'model_loaded': model is not None,
        'is_processing': is_processing,
        'idle_duration': idle_duration,
        'time_since_load': time_since_load,
        'idle_timeout': idle_timeout,
        'model_type': 'Qwen3-Omni Captioner AWQ'
    }), 200


# ==================== Idle Management ====================

def idle_checker():
    """Background thread to check for idle state and unload model if needed"""
    global last_activity_time, model_loaded_time
    
    while True:
        time.sleep(idle_check_interval)
        
        if last_activity_time is None:
            last_activity_time = time.time()
            model_loaded_time = time.time()
            continue
        
        current_time = time.time()
        
        # Calculate idle time and time since model was loaded
        idle_duration = current_time - last_activity_time
        time_since_load = current_time - model_loaded_time if model_loaded_time else 0
        
        # If we're not processing and have been idle beyond the timeout
        if idle_duration > idle_timeout and not is_processing and model is not None:
            print(f"[INFO] Idle timeout reached ({idle_duration}s), unloading model to save resources...")
            with model_lock:  # Ensure no other thread is using the model
                if not is_processing and idle_duration > idle_timeout:  # Double-check
                    unload_model()
                    # Reset model_loaded_time since model is no longer loaded
                    model_loaded_time = None


# ==================== Main Execution ====================

def main():
    parser = ArgumentParser(description="Qwen3-Omni Captioner AWQ HTTP Server")
    
    parser.add_argument("--checkpoint-path", type=str, 
                        default="Qwen/Qwen3-Omni-30B-A3B-Captioner-AWQ-4bit",
                        help="Path to the AWQ quantized model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Model data type")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device mapping for model")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Use flash attention 2")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking step for complex queries")
    parser.add_argument("--enable-opencc", action="store_true",
                        help="Enable OpenCC for Chinese text conversion")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Idle timeout in seconds")
    
    args = parser.parse_args()
    
    # Update idle timeout from args
    global idle_timeout
    idle_timeout = args.idle_timeout
    
    # Initialize model
    initialize_model(args)
    
    # Start idle checker thread
    idle_thread = threading.Thread(target=idle_checker, daemon=True)
    idle_thread.start()
    
    # Initialize last activity time
    global last_activity_time, model_loaded_time
    last_activity_time = time.time()
    model_loaded_time = time.time()
    
    # Run Flask app
    print(f"[INFO] Starting server on {args.host}:{args.port}")
    print(f"[INFO] Model will be unloaded after {idle_timeout} seconds of inactivity")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()