#!/usr/bin/env python
# qwen3_omni.py - Serve script for Qwen3-Omni model

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from flask import Flask, request, jsonify
import soundfile as sf
import base64
from io import BytesIO

# Register Qwen3OmniMoe models to handle the custom architecture
try:
    from register_qwen3_omni import register_all
    register_all()
except ImportError:
    print("[ERROR] Could not import register_qwen3_omni.")
    raise

app = Flask(__name__)

# Global variables for model and processor
model = None
processor = None


def setup_model(args):
    global model, processor
    
    print(f"Loading model from: {args.checkpoint_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.float16 if not args.float32 else torch.float32,
            device_map="auto",
            attn_implementation="flash_attention_2" if args.flash_attn2 else None,
            trust_remote_code=True
        )
        
        # Load processor - using dynamic loading as the specific processor might not be available
        try:
            processor = AutoProcessor.from_pretrained(args.checkpoint_path, trust_remote_code=True)
        except (AttributeError, ValueError):
            # If AutoProcessor fails, try to get the specific processor class dynamically
            from transformers import AutoTokenizer
            # Fallback to tokenizer
            processor = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        print("[INFO] This error may occur if the transformers library doesn't recognize the model architecture.")
        print("[INFO] You may need to update transformers with the command: pip install git+https://github.com/huggingface/transformers.git")
        raise e


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        text_input = data.get('text', '')
        audio_input = data.get('audio', None)  # Base64 encoded audio
        image_input = data.get('image', None)  # Base64 encoded image
        
        # Create conversation format
        conversation = [{"role": "user", "content": []}]
        
        # Add text content if provided
        if text_input:
            conversation[0]["content"].append({"type": "text", "text": text_input})
            
        # Add image content if provided
        if image_input:
            # For now, just append the image data - in real usage you'd save it temporarily
            conversation[0]["content"].append({"type": "image", "image": image_input})
            
        # Add audio content if provided
        if audio_input:
            conversation[0]["content"].append({"type": "audio", "audio": audio_input})
        
        # Process the conversation
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # For now, just generate text response - real implementation would handle multimodal inputs
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        
        # Generate response
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        # Decode response
        response = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # If the model supports audio generation, we would handle that here
        audio_output = None
        
        return jsonify({
            'text': response,
            'audio': audio_output  # In real implementation, this would be the generated audio
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/v1/chat/completions', methods=['POST'])
def openai_compatible_endpoint():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        # Convert OpenAI format to Qwen format
        conversation = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if isinstance(content, str):
                conversation.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                conversation.append({
                    "role": role,
                    "content": content
                })
        
        # Process the conversation
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # For now, just generate text response - real implementation would handle multimodal inputs
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        
        # Generate response
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        # Decode response
        response = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return jsonify({
            'id': 'chatcmpl-' + str(hash(response))[:8],
            'object': 'chat.completion',
            'created': 1677610602,
            'model': 'qwen3-omni',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response,
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(inputs["input_ids"][0]),
                'completion_tokens': len(output_ids[0]) - len(inputs["input_ids"][0]),
                'total_tokens': len(output_ids[0])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni HTTP Server")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Use flash attention 2")
    parser.add_argument("--float32", action="store_true",
                        help="Use float32 instead of float16")
    parser.add_argument("--audio-only", action="store_true",
                        help="Audio only mode")
    parser.add_argument("--segment-duration", type=int, default=60,
                        help="Audio segment duration")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--idle-timeout", type=int, default=600,
                        help="Idle timeout in seconds")
    parser.add_argument("--local-model", action="store_true",
                        help="Use local model")
    
    args = parser.parse_args()
    
    # Setup model
    setup_model(args)
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()