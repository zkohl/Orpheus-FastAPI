# Updating Packages for Zero-Shot Voice Cloning

The Orpheus model uses a newer configuration format that requires updated packages.

## Update Commands

Run these commands to update to the compatible versions:

```bash
# Windows (PowerShell)
pip install transformers==4.44.0 accelerate==0.33.0 protobuf sentencepiece --upgrade

# Or if you're in a virtual environment:
python -m pip install transformers==4.44.0 accelerate==0.33.0 protobuf sentencepiece --upgrade
```

## Why This Update?

The error:
```
`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, got {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
```

This indicates the model uses the newer Llama3 RoPE (Rotary Position Embedding) scaling format, which was introduced in transformers 4.40+. The older version (4.36.0) doesn't understand this format.

## After Updating

Once you've updated the packages, restart the server:
```bash
python app.py
```

The model should now load successfully with the updated transformers library that understands the Llama3 configuration format.