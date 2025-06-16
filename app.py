# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import os
import time
import asyncio
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Function to ensure .env file exists
def ensure_env_file_exists():
    """Create a .env file from defaults and OS environment variables"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        try:
            # 1. Create default env dictionary from .env.example
            default_env = {}
            with open(".env.example", "r") as example_file:
                for line in example_file:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        default_env[key] = line.split("=", 1)[1].strip()

            # 2. Override defaults with Docker environment variables if they exist
            final_env = default_env.copy()
            for key in default_env:
                if key in os.environ:
                    final_env[key] = os.environ[key]

            # 3. Write dictionary to .env file in env format
            with open(".env", "w") as env_file:
                for key, value in final_env.items():
                    env_file.write(f"{key}={value}\n")
                    
            print("✅ Created default .env file from .env.example and environment variables.")
        except Exception as e:
            print(f"⚠️ Error creating default .env file: {e}")

# Ensure .env file exists before loading environment variables
ensure_env_file_exists()

# Load environment variables from .env file
load_dotenv(override=True)

from fastapi import FastAPI, Request, Form, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import tempfile
import shutil

from tts_engine import generate_speech_from_api, AVAILABLE_VOICES, DEFAULT_VOICE, VOICE_TO_LANGUAGE, AVAILABLE_LANGUAGES

# Import voice cloning components
try:
    from orpheus_voice_clone import VoiceCloneModel, ModelConfig
    VOICE_CLONE_AVAILABLE = True
    print("✅ Voice cloning module loaded successfully")
except ImportError as e:
    VOICE_CLONE_AVAILABLE = False
    print(f"⚠️ Voice cloning not available: {e}")

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0"
)

# We'll use FastAPI's built-in startup complete mechanism
# The log message "INFO:     Application startup complete." indicates
# that the application is ready

# Load default voice clone config at startup
if VOICE_CLONE_AVAILABLE:
    from orpheus_voice_clone import ModelConfig
    default_voice_clone_config = ModelConfig(
        huggingface_token=os.environ.get("HF_TOKEN")
    )
    print("\n🎤 Default Zero-Shot Voice Clone Configuration:")
    print(f"   Max New Tokens: {default_voice_clone_config.max_new_tokens}")
    print(f"   Temperature: {default_voice_clone_config.temperature}")
    print(f"   Top P: {default_voice_clone_config.top_p}")
    print(f"   Repetition Penalty: {default_voice_clone_config.repetition_penalty}")
    print(f"   Sample Rate: {default_voice_clone_config.sample_rate} Hz")
    print(f"   Device: {default_voice_clone_config.device}")
    print("")

# Initialize voice clone model (singleton pattern)
voice_clone_model = None
default_voice_clone_config = None

async def get_voice_clone_model(config: Optional[ModelConfig] = None):
    """Get or initialize the voice clone model with optional config overrides"""
    global voice_clone_model, default_voice_clone_config
    if not VOICE_CLONE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Voice cloning feature is not available")
    
    if voice_clone_model is None:
        print("🎤 Initializing voice clone model...")
        # Get HuggingFace token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=503,
                detail="HF_TOKEN not configured. Please set the HuggingFace token in the .env file."
            )
        
        # Use provided config or default
        if config is None:
            config = default_voice_clone_config if default_voice_clone_config else ModelConfig(
                huggingface_token=hf_token
            )
        
        try:
            voice_clone_model = VoiceCloneModel(config)
            # Warmup the model for better performance
            voice_clone_model.warmup()
            print("✅ Voice clone model initialized and warmed up")
        except Exception as e:
            print(f"❌ Failed to initialize voice clone model: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to initialize voice clone model: {str(e)}")
    
    return voice_clone_model

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount directories for serving files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

class APIResponse(BaseModel):
    status: str
    voice: str
    output_file: str
    generation_time: float

# OpenAI-compatible API endpoint
@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    
    For longer texts (>1000 characters), batched generation is used
    to improve reliability and avoid truncation issues.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    # Check if we should use batched generation
    use_batching = len(request.input) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(request.input)} characters)")
    
    # Generate speech with automatic batching for long texts
    start = time.time()
    generate_speech_from_api(
        prompt=request.input,
        voice=request.voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000  # Process in ~1000 character chunks (roughly 1 paragraph)
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    # Return audio file
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request.voice}_{timestamp}.wav"
    )

@app.post("/v1/audio/speech/zero-shot")
async def create_zero_shot_speech(
    text: str = Form(..., description="Text to be spoken with the cloned voice"),
    voice_transcript: str = Form(..., description="Transcript of what's said in the voice sample"),
    voice_audio: UploadFile = File(..., description="Voice sample audio file (WAV format)"),
    max_new_tokens: Optional[int] = Form(None, description="Maximum number of new tokens to generate"),
    temperature: Optional[float] = Form(None, description="Temperature for generation (0.0-1.0)"),
    top_p: Optional[float] = Form(None, description="Top-p sampling parameter (0.0-1.0)"),
    repetition_penalty: Optional[float] = Form(None, description="Repetition penalty parameter")
):
    """
    Generate speech using zero-shot voice cloning.
    
    This endpoint clones a voice from a provided audio sample and uses it to speak new text.
    The voice cloning runs directly on this server, not through the external LLM API.
    
    Parameters:
    - text: The text you want the cloned voice to speak
    - voice_transcript: Accurate transcript of what's being said in the voice sample
    - voice_audio: Audio file containing the voice sample (WAV format recommended)
    - max_new_tokens: (Optional) Maximum number of new tokens to generate
    - temperature: (Optional) Temperature for generation (0.0-1.0)
    - top_p: (Optional) Top-p sampling parameter (0.0-1.0)
    - repetition_penalty: (Optional) Repetition penalty parameter
    
    Example usage:
    ```bash
    curl -X POST https://orpheus.zkohl.net/v1/audio/speech/zero-shot \\
      -F 'text=I finally got into the university of my dreams!' \\
      -F 'voice_transcript=Okay, you are relentless...' \\
      -F 'voice_audio=@/path/to/voice_sample.wav' \\
      -F 'temperature=0.7' \\
      -F 'max_new_tokens=1000' \\
      --output output.wav
    ```
    """
    # Validate inputs
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' parameter")
    if not voice_transcript:
        raise HTTPException(status_code=400, detail="Missing 'voice_transcript' parameter")
    if not voice_audio:
        raise HTTPException(status_code=400, detail="Missing 'voice_audio' file")
    
    # Check file type
    if not voice_audio.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400,
            detail="Invalid audio format. Supported formats: WAV, MP3, M4A, FLAC, OGG"
        )
    
    # Create config with client overrides if provided
    config_overrides = {}
    if max_new_tokens is not None:
        config_overrides['max_new_tokens'] = max_new_tokens
    if temperature is not None:
        config_overrides['temperature'] = temperature
    if top_p is not None:
        config_overrides['top_p'] = top_p
    if repetition_penalty is not None:
        config_overrides['repetition_penalty'] = repetition_penalty
    
    # Print configuration being used
    if config_overrides:
        print("🔧 Using custom generation parameters:")
        for key, value in config_overrides.items():
            print(f"   {key}: {value}")
    
    # Get the voice clone model
    model = await get_voice_clone_model()
    
    # Create temporary file for the uploaded audio
    temp_audio_path = None
    temp_output_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            temp_audio_path = tmp_file.name
            shutil.copyfileobj(voice_audio.file, tmp_file)
        
        print(f"📤 Received voice sample: {voice_audio.filename} ({voice_audio.size} bytes)")
        print(f"📝 Text to generate: {text[:100]}...")
        
        # Generate unique output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"zero_shot_{timestamp}.wav"
        output_path = f"outputs/{output_filename}"
        
        # Perform voice cloning
        start_time = time.time()
        print("🎙️ Starting zero-shot voice cloning...")
        
        try:
            # Use the voice clone model to generate audio with optional parameters
            generated_paths = model.clone_voice(
                voice_sample_path=temp_audio_path,
                voice_transcript=voice_transcript,
                target_texts=[text],  # Process as single text
                output_dir="outputs",
                **config_overrides  # Pass generation parameters
            )
            
            if not generated_paths:
                raise HTTPException(status_code=500, detail="Voice cloning failed to generate audio")
            
            # Rename the generated file to our desired output path
            if os.path.exists(generated_paths[0]):
                shutil.move(generated_paths[0], output_path)
            else:
                raise HTTPException(status_code=500, detail="Generated audio file not found")
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            print(f"✅ Voice cloning completed in {generation_time:.2f} seconds")
            print(f"📦 Output saved to: {output_path}")
            
            # Return the generated audio file
            return FileResponse(
                path=output_path,
                media_type="audio/wav",
                filename=output_filename,
                headers={
                    "X-Generation-Time": str(generation_time),
                    "X-Voice-Clone": "zero-shot"
                }
            )
            
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        except Exception as e:
            print(f"❌ Voice cloning error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {e}")

@app.get("/v1/audio/voices")
async def list_voices():
    """Return list of available voices"""
    if not AVAILABLE_VOICES or len(AVAILABLE_VOICES) == 0:
        raise HTTPException(status_code=404, detail="No voices available")
    return JSONResponse(
        content={
            "status": "ok",
            "voices": AVAILABLE_VOICES,
            "zero_shot_available": VOICE_CLONE_AVAILABLE
        }
    )

# Legacy API endpoint for compatibility
@app.post("/speak")
async def speak(request: Request):
    """Legacy endpoint for compatibility with existing clients"""
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)

    if not text:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing 'text'"}
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Check if we should use batched generation for longer texts
    use_batching = len(text) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(text)} characters)")
    
    # Generate speech with batching for longer texts
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    end = time.time()
    generation_time = round(end - start, 2)

    return JSONResponse(content={
        "status": "ok",
        "voice": voice,
        "output_file": output_path,
        "generation_time": generation_time
    })

# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to web UI"""
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request, 
            "voices": AVAILABLE_VOICES,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.get("/web/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Main web UI for TTS generation"""
    # Get current config for the Web UI
    config = get_current_config()
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request, 
            "voices": AVAILABLE_VOICES, 
            "config": config,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.get("/get_config")
async def get_config():
    """Get current configuration from .env file or defaults"""
    config = get_current_config()
    return JSONResponse(content=config)

@app.post("/save_config")
async def save_config(request: Request):
    """Save configuration to .env file"""
    data = await request.json()
    
    # Convert values to proper types
    for key, value in data.items():
        if key in ["ORPHEUS_MAX_TOKENS", "ORPHEUS_API_TIMEOUT", "ORPHEUS_PORT", "ORPHEUS_SAMPLE_RATE"]:
            try:
                data[key] = str(int(value))
            except (ValueError, TypeError):
                pass
        elif key in ["ORPHEUS_TEMPERATURE", "ORPHEUS_TOP_P"]:  # Removed ORPHEUS_REPETITION_PENALTY since it's hardcoded now
            try:
                data[key] = str(float(value))
            except (ValueError, TypeError):
                pass
    
    # Write configuration to .env file
    with open(".env", "w") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")
    
    return JSONResponse(content={"status": "ok", "message": "Configuration saved successfully. Restart server to apply changes."})

@app.post("/restart_server")
async def restart_server():
    """Restart the server by touching a file that triggers Uvicorn's reload"""
    import threading
    
    def touch_restart_file():
        # Wait a moment to let the response get back to the client
        time.sleep(0.5)
        
        # Create or update restart.flag file to trigger reload
        restart_file = "restart.flag"
        with open(restart_file, "w") as f:
            f.write(str(time.time()))
            
        print("🔄 Restart flag created, server will reload momentarily...")
    
    # Start the touch operation in a separate thread
    threading.Thread(target=touch_restart_file, daemon=True).start()
    
    # Return success response
    return JSONResponse(content={"status": "ok", "message": "Server is restarting. Please wait a moment..."})

def get_current_config():
    """Read current configuration from .env.example and .env files"""
    # Default config from .env.example
    default_config = {}
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    default_config[key] = value
    
    # Current config from .env
    current_config = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    current_config[key] = value
    
    # Merge configs, with current taking precedence
    config = {**default_config, **current_config}
    
    # Add current environment variables
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
    return config

@app.post("/web/", response_class=HTMLResponse)
async def generate_from_web(
    request: Request,
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE)
):
    """Handle form submission from web UI"""
    if not text:
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": "Please enter some text.",
                "voices": AVAILABLE_VOICES,
                "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
                "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
            }
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Check if we should use batched generation for longer texts
    use_batching = len(text) > 1000
    if use_batching:
        print(f"Using batched generation for long text from web form ({len(text)} characters)")
    
    # Generate speech with batching for longer texts
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "success": True,
            "text": text,
            "voice": voice,
            "output_file": output_path,
            "generation_time": generation_time,
            "voices": AVAILABLE_VOICES,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required settings
    required_settings = ["ORPHEUS_HOST", "ORPHEUS_PORT"]
    missing_settings = [s for s in required_settings if s not in os.environ]
    if missing_settings:
        print(f"⚠️ Missing environment variable(s): {', '.join(missing_settings)}")
        print("   Using fallback values for server startup.")
    
    # Get host and port from environment variables with better error handling
    try:
        host = os.environ.get("ORPHEUS_HOST")
        if not host:
            print("⚠️ ORPHEUS_HOST not set, using 0.0.0.0 as fallback")
            host = "0.0.0.0"
    except Exception:
        print("⚠️ Error reading ORPHEUS_HOST, using 0.0.0.0 as fallback")
        host = "0.0.0.0"
        
    try:
        port = int(os.environ.get("ORPHEUS_PORT", "5005"))
    except (ValueError, TypeError):
        print("⚠️ Invalid ORPHEUS_PORT value, using 5005 as fallback")
        port = 5005
    
    print(f"🔥 Starting Orpheus-FASTAPI Server on {host}:{port}")
    print(f"💬 Web UI available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"📖 API docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    
    # Read current API_URL for user information
    api_url = os.environ.get("ORPHEUS_API_URL")
    if not api_url:
        print("⚠️ ORPHEUS_API_URL not set. Please configure in .env file before generating speech.")
    else:
        print(f"🔗 Using LLM inference server at: {api_url}")
        
    # Include restart.flag in the reload_dirs to monitor it for changes
    extra_files = ["restart.flag"] if os.path.exists("restart.flag") else []
    
    # Start with reload enabled to allow automatic restart when restart.flag changes
    uvicorn.run("app:app", host=host, port=port, reload=True, reload_dirs=["."], reload_includes=["*.py", "*.html", "restart.flag"])