#!/usr/bin/env python3
"""
Example usage of the Orpheus Voice Clone model.

This demonstrates how to use the refactored voice cloning system
for generating speech with a cloned voice.
"""

import os
from orpheus_voice_clone import VoiceCloneModel, ModelConfig


def main():
    """Main example function."""
    
    # Configure the model
    config = ModelConfig(
        huggingface_token=os.environ.get("HF_TOKEN"),
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # Initialize the model (this will be done once in a server context)
    print("Initializing voice clone model...")
    model = VoiceCloneModel(config)
    
    # Warmup the model (optional but recommended for servers)
    model.warmup()
    
    # Define inputs
    voice_sample_path = "input_audio/zach.wav"
    voice_transcript = (
        "Okay, you are relentless, I like it. They were weighing up a few options, "
        "but Pinecone won out because of its speed and scalability, especially with "
        "the kind of volume of data we're dealing with. It lets me pull up relevant "
        "info pretty quickly, which is good, because nobody wants to wait ages for "
        "an AI to remember something."
    )
    
    # Define what the model should say
    target_texts = [
        "I finally got into the university of my dreams! I can't believe all this hard work actually paid off!",
        # Add more texts as needed
    ]
    
    # Generate audio
    print("\nGenerating audio samples...")
    try:
        output_paths = model.clone_voice(
            voice_sample_path=voice_sample_path,
            voice_transcript=voice_transcript,
            target_texts=target_texts,
            output_dir="output_audio"
        )
        
        print(f"\n✅ Successfully generated {len(output_paths)} audio files:")
        for path in output_paths:
            print(f"   - {path}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the input audio file exists.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


def batch_example():
    """Example of batch processing with the same voice."""
    
    config = ModelConfig()
    model = VoiceCloneModel(config)
    
    voice_sample = "input_audio/zach.wav"
    transcript = "Your voice sample transcript here..."
    
    # Multiple batches of texts
    text_batches = [
        ["Hello, this is batch 1, text 1.", "This is batch 1, text 2."],
        ["Hello, this is batch 2, text 1.", "This is batch 2, text 2."],
    ]
    
    # Process all batches efficiently
    all_outputs = model.process_batch(
        voice_sample_path=voice_sample,
        voice_transcript=transcript,
        target_texts_batch=text_batches,
        output_base_dir="batch_output"
    )
    
    print(f"Generated {sum(len(batch) for batch in all_outputs)} total audio files")


if __name__ == "__main__":
    main()
    
    # Uncomment to run batch example
    # batch_example()