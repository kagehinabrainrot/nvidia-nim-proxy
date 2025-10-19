from flask import Flask, request, jsonify, Response
import requests
import json
import time

app = Flask(__name__)

# Configuration
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-r1-0528"

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models in OpenAI format"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "meta/llama-3.1-70b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia"
            },
            {
                "id": "meta/llama-3.1-8b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia"
            }
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Proxy chat completions from OpenAI format to NVIDIA NIM"""
    try:
        data = request.json
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', DEFAULT_MODEL)
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Build NVIDIA NIM request
        nim_request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            return handle_streaming(nim_request, headers)
        else:
            return handle_non_streaming(nim_request, headers)
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": 500
            }
        }), 500

def handle_non_streaming(nim_request, headers):
    """Handle non-streaming responses"""
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers=headers,
        json=nim_request
    )
    
    if response.status_code != 200:
        return jsonify({
            "error": {
                "message": response.text,
                "type": "nvidia_api_error",
                "code": response.status_code
            }
        }), response.status_code
    
    nim_response = response.json()
    
    # Convert NVIDIA response to OpenAI format
    openai_response = {
        "id": nim_response.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": nim_response.get("model", nim_request["model"]),
        "choices": nim_response.get("choices", []),
        "usage": nim_response.get("usage", {})
    }
    
    return jsonify(openai_response)

def handle_streaming(nim_request, headers):
    """Handle streaming responses"""
    def generate():
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nim_request,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    yield decoded_line + '\n\n'
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("Starting OpenAI-compatible proxy for NVIDIA NIM...")
    print(f"Server will run on http://0.0.0.0:5000")
    print(f"Use this as your API endpoint in Janitor AI")
    app.run(host='0.0.0.0', port=5000, debug=False)
