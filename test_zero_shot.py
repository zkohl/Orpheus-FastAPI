#!/usr/bin/env python3
"""
Test script for zero-shot voice cloning with Orpheus FastAPI

This script demonstrates how to use the zero-shot voice cloning endpoint.
"""

import requests
import sys
import os

def test_zero_shot_cloning(
    api_url="http://localhost:8000",
    text="I finally got into the university of my dreams! I can't believe all this hard work actually paid off!",
    voice_audio_path="test_voice.wav",
    voice_transcript="Hello, this is a test of the voice cloning system."
):
    """Test the zero-shot voice cloning endpoint"""
    
    endpoint = f"{api_url}/v1/audio/speech/zero-shot"
    
    # Check if voice audio file exists
    if not os.path.exists(voice_audio_path):
        print(f"Error: Voice audio file '{voice_audio_path}' not found!")
        print("Please provide a WAV file with clear speech (3-10 seconds)")
        return False
    
    # Prepare the request
    try:
        with open(voice_audio_path, 'rb') as audio_file:
            files = {
                'voice_audio': (os.path.basename(voice_audio_path), audio_file, 'audio/wav')
            }
            data = {
                'text': text,
                'voice_transcript': voice_transcript
            }
            
            print(f"Sending request to {endpoint}")
            print(f"Text: {text}")
            print(f"Voice transcript: {voice_transcript}")
            print(f"Voice audio: {voice_audio_path}")
            
            # Send the request
            response = requests.post(endpoint, files=files, data=data)
            
            if response.status_code == 200:
                # Save the output audio
                output_path = "zero_shot_output.wav"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"\n✅ Success! Generated audio saved to: {output_path}")
                print(f"Generation time: {response.headers.get('X-Generation-Time', 'N/A')} seconds")
                return True
            else:
                print(f"\n❌ Error: {response.status_code}")
                print(f"Details: {response.text}")
                return False
                
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        return False

def main():
    """Main function"""
    print("Zero-Shot Voice Cloning Test")
    print("=" * 40)
    
    # You can customize these parameters
    if len(sys.argv) > 1:
        voice_audio_path = sys.argv[1]
    else:
        voice_audio_path = "test_voice.wav"
    
    if len(sys.argv) > 2:
        voice_transcript = sys.argv[2]
    else:
        voice_transcript = "Hello, this is a test of the voice cloning system."
    
    if len(sys.argv) > 3:
        text = sys.argv[3]
    else:
        text = "I finally got into the university of my dreams! I can't believe all this hard work actually paid off!"
    
    # Test the endpoint
    success = test_zero_shot_cloning(
        voice_audio_path=voice_audio_path,
        voice_transcript=voice_transcript,
        text=text
    )
    
    if not success:
        print("\nUsage: python test_zero_shot.py [voice_audio.wav] [voice_transcript] [text_to_synthesize]")
        print("\nExample:")
        print('  python test_zero_shot.py my_voice.wav "Hello world" "This is the text I want to synthesize"')

if __name__ == "__main__":
    main()