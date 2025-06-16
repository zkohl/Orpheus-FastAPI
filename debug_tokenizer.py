#!/usr/bin/env python3
"""
Debug script for tokenizer loading issues with Orpheus Voice Clone model.
"""

import os
import sys
from dotenv import load_dotenv
from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast
import torch

# Load environment variables
load_dotenv(override=True)

def test_tokenizer_loading():
    """Test different methods of loading the tokenizer."""
    
    # Get the model name from the config (default from orpheus_voice_clone)
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    print(f"Testing tokenizer loading for model: {model_name}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {import_transformers_version()}")
    print("-" * 60)
    
    # Test 1: Standard AutoTokenizer
    print("\n1. Testing AutoTokenizer.from_pretrained()...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Success with AutoTokenizer!")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed with AutoTokenizer: {type(e).__name__}: {e}")
    
    # Test 2: AutoTokenizer with use_fast=False
    print("\n2. Testing AutoTokenizer with use_fast=False...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("✅ Success with AutoTokenizer (slow)!")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed with AutoTokenizer (slow): {type(e).__name__}: {e}")
    
    # Test 3: LlamaTokenizer directly
    print("\n3. Testing LlamaTokenizer.from_pretrained()...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        print("✅ Success with LlamaTokenizer!")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed with LlamaTokenizer: {type(e).__name__}: {e}")
    
    # Test 4: LlamaTokenizerFast directly
    print("\n4. Testing LlamaTokenizerFast.from_pretrained()...")
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
        print("✅ Success with LlamaTokenizerFast!")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed with LlamaTokenizerFast: {type(e).__name__}: {e}")
    
    # Test 5: Try with trust_remote_code
    print("\n5. Testing with trust_remote_code=True...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Success with trust_remote_code!")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed with trust_remote_code: {type(e).__name__}: {e}")
    
    # Test 6: Clear cache and retry
    print("\n6. Testing after clearing cache...")
    try:
        # Clear the cache for this specific model
        from transformers.utils import TRANSFORMERS_CACHE
        import shutil
        
        cache_dir = os.path.join(TRANSFORMERS_CACHE, f"models--{model_name.replace('/', '--')}")
        if os.path.exists(cache_dir):
            print(f"   Found cache at: {cache_dir}")
            response = input("   Clear cache and re-download? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(cache_dir)
                print("   Cache cleared. Retrying...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("✅ Success after clearing cache!")
                print(f"   Tokenizer type: {type(tokenizer)}")
                print(f"   Vocab size: {tokenizer.vocab_size}")
            else:
                print("   Skipping cache clear.")
        else:
            print("   No cache found for this model.")
    except Exception as e:
        print(f"❌ Failed after cache clear: {type(e).__name__}: {e}")

def import_transformers_version():
    """Get transformers version."""
    try:
        import transformers
        return transformers.__version__
    except:
        return "Unknown"

def suggest_fix():
    """Suggest fixes based on the test results."""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES:")
    print("=" * 60)
    print("""
1. Update transformers to the latest version:
   pip install --upgrade transformers

2. Clear the model cache:
   rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct

3. Try using a different tokenizer loading method in the code:
   - Use use_fast=False parameter
   - Use LlamaTokenizer directly instead of AutoTokenizer
   - Add trust_remote_code=True parameter

4. Check if the model requires a specific transformers version:
   - The model card might specify version requirements

5. Ensure all dependencies are compatible:
   pip install --upgrade transformers accelerate sentencepiece protobuf
""")

if __name__ == "__main__":
    test_tokenizer_loading()
    suggest_fix()