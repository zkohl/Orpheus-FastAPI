"""Tokenization utilities for audio and text processing."""

from typing import List, Tuple

import torch
import numpy as np


class AudioTokenizer:
    """Handles audio tokenization for SNAC model."""
    
    def __init__(self, snac_model):
        """
        Initialize audio tokenizer.
        
        Args:
            snac_model: Initialized SNAC model instance
        """
        self.snac_model = snac_model
        self.base_offset = 128266
        self.layer_offsets = [0, 4096, 8192, 12288, 16384, 20480, 24576]
    
    def tokenize_audio(self, waveform: np.ndarray) -> List[int]:
        """
        Convert audio waveform to token sequence.
        
        Args:
            waveform: Audio waveform as numpy array
        
        Returns:
            List of audio tokens
        """
        print(f"Tokenizing audio waveform with shape: {waveform.shape}")
        
        # Prepare waveform tensor
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
        waveform_tensor = waveform_tensor.to(dtype=torch.float32).unsqueeze(0)
        
        print(f"Waveform tensor shape: {waveform_tensor.shape}")
        
        # Encode audio
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform_tensor)
        
        print(f"Encoded audio - Layer shapes: {[c.shape for c in codes]}")
        
        # Interleave codes according to SNAC format
        all_codes = []
        for i in range(codes[0].shape[1]):
            # Layer 1
            all_codes.append(codes[0][0][i].item() + self.base_offset)
            # Layer 2 (first)
            all_codes.append(codes[1][0][2*i].item() + self.base_offset + self.layer_offsets[1])
            # Layer 3 (four tokens)
            all_codes.append(codes[2][0][4*i].item() + self.base_offset + self.layer_offsets[2])
            all_codes.append(codes[2][0][(4*i)+1].item() + self.base_offset + self.layer_offsets[3])
            # Layer 2 (second)
            all_codes.append(codes[1][0][(2*i)+1].item() + self.base_offset + self.layer_offsets[4])
            # Layer 3 (remaining two)
            all_codes.append(codes[2][0][(4*i)+2].item() + self.base_offset + self.layer_offsets[5])
            all_codes.append(codes[2][0][(4*i)+3].item() + self.base_offset + self.layer_offsets[6])
        
        print(f"Generated {len(all_codes)} audio tokens")
        print(f"Token range: {min(all_codes)} to {max(all_codes)}")
        
        return all_codes
    
    def detokenize_audio(self, token_list: List[int]) -> torch.Tensor:
        """
        Convert token sequence back to audio waveform.
        
        Args:
            token_list: List of audio tokens
        
        Returns:
            Audio waveform as torch tensor
        """
        print(f"Detokenizing {len(token_list)} audio tokens")
        
        if len(token_list) < 7:
            raise ValueError(f"Token list too short: {len(token_list)} tokens (minimum 7 required)")
        
        # Remove base offset
        adjusted_tokens = [t - self.base_offset for t in token_list]
        
        # Redistribute tokens to layers
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        num_groups = (len(adjusted_tokens) + 1) // 7
        for i in range(num_groups):
            base_idx = 7 * i
            if base_idx >= len(adjusted_tokens):
                break
            
            # Extract tokens for each layer
            layer_1.append(adjusted_tokens[base_idx])
            
            if base_idx + 1 < len(adjusted_tokens):
                layer_2.append(adjusted_tokens[base_idx + 1] - self.layer_offsets[1])
            
            for j, offset_idx in enumerate([2, 3, 5, 6]):
                if base_idx + offset_idx < len(adjusted_tokens):
                    layer_3.append(adjusted_tokens[base_idx + offset_idx] - self.layer_offsets[offset_idx])
            
            if base_idx + 4 < len(adjusted_tokens):
                layer_2.append(adjusted_tokens[base_idx + 4] - self.layer_offsets[4])
        
        # Convert to tensors
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
        
        # Decode to audio
        with torch.no_grad():
            audio_hat = self.snac_model.decode(codes)
        
        # Ensure the tensor is detached from computation graph
        if audio_hat.requires_grad:
            audio_hat = audio_hat.detach()
        
        return audio_hat


def prepare_prompt_tokens(
    text_tokenizer,
    audio_tokens: List[int],
    voice_prompt: str,
    target_texts: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare input tokens for voice cloning generation.
    
    Args:
        text_tokenizer: Text tokenizer instance
        audio_tokens: Tokenized audio
        voice_prompt: Transcript of the voice sample
        target_texts: List of texts to generate
    
    Returns:
        Tuple of (input_ids, attention_mask)
    """
    print(f"Preparing prompts - Audio tokens: {len(audio_tokens)}, Voice prompt: '{voice_prompt}', Target texts: {len(target_texts)}")
    
    # Special tokens
    SOH = 128259  # Start of human
    SOT = 128257  # Start of text
    EOT = 128009  # End of text
    EOH = 128260  # End of human
    SOA = 128261  # Start of AI
    EOS = 128258  # End of speech
    SOS = 128262  # Start of speech
    PAD = 128263  # Padding token
    
    # Tokenize voice prompt
    prompt_tokens = text_tokenizer(voice_prompt, return_tensors="pt")["input_ids"]
    
    # Build zero-shot prompt
    # Format: SOH SOT Text EOT EOH SOA audio_tokens EOS SOS
    zeroprompt_input_ids = torch.cat([
        torch.tensor([[SOH]], dtype=torch.int64),
        prompt_tokens,
        torch.tensor([[EOT, EOH, SOA, SOT]], dtype=torch.int64),
        torch.tensor([audio_tokens], dtype=torch.int64),
        torch.tensor([[EOS, SOS]], dtype=torch.int64)
    ], dim=1)
    
    # Prepare all prompts
    all_input_ids = []
    for target_text in target_texts:
        target_tokens = text_tokenizer(target_text, return_tensors="pt")["input_ids"]
        full_input_ids = torch.cat([
            zeroprompt_input_ids,
            torch.tensor([[SOH]], dtype=torch.int64),
            target_tokens,
            torch.tensor([[EOT]], dtype=torch.int64)
        ], dim=1)
        all_input_ids.append(full_input_ids)
    
    # Pad to same length
    max_length = max(ids.shape[1] for ids in all_input_ids)
    
    padded_tensors = []
    attention_masks = []
    
    for input_ids in all_input_ids:
        padding_length = max_length - input_ids.shape[1]
        
        # Left-pad with PAD tokens
        padded = torch.cat([
            torch.full((1, padding_length), PAD, dtype=torch.int64),
            input_ids
        ], dim=1)
        
        # Create attention mask (0 for padding, 1 for real tokens)
        mask = torch.cat([
            torch.zeros((1, padding_length), dtype=torch.int64),
            torch.ones((1, input_ids.shape[1]), dtype=torch.int64)
        ], dim=1)
        
        padded_tensors.append(padded)
        attention_masks.append(mask)
    
    # Stack all sequences
    input_ids = torch.cat(padded_tensors, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_mask


def extract_generated_audio_tokens(
    generated_ids: torch.Tensor,
    start_token: int = 128257,
    end_token: int = 128258
) -> List[List[int]]:
    """
    Extract audio tokens from generated sequences.
    
    Args:
        generated_ids: Generated token sequences
        start_token: Token marking start of audio
        end_token: Token to remove from output
    
    Returns:
        List of audio token sequences
    """
    print(f"Extracting audio tokens from sequences of shape: {generated_ids.shape}")
    print(f"Looking for start token: {start_token}, end token: {end_token}")
    
    token_lists = []
    
    for idx, sequence in enumerate(generated_ids):
        # Debug: print first and last 20 tokens
        print(f"Sequence {idx} - First 20 tokens: {sequence[:20].tolist()}")
        print(f"Sequence {idx} - Last 20 tokens: {sequence[-20:].tolist()}")
        
        # Find last occurrence of start token
        start_indices = (sequence == start_token).nonzero(as_tuple=True)[0]
        
        if len(start_indices) > 0:
            last_start_idx = start_indices[-1].item()
            print(f"Found start token at index: {last_start_idx}")
            cropped = sequence[last_start_idx + 1:]
        else:
            print(f"Warning: No start token found in sequence {idx}")
            cropped = sequence
        
        # Remove end tokens
        audio_tokens = cropped[cropped != end_token].tolist()
        print(f"Extracted {len(audio_tokens)} tokens after removing end tokens")
        
        # Ensure length is multiple of 7 for proper decoding
        trimmed_length = (len(audio_tokens) // 7) * 7
        if trimmed_length != len(audio_tokens):
            print(f"Trimming from {len(audio_tokens)} to {trimmed_length} tokens (multiple of 7)")
        audio_tokens = audio_tokens[:trimmed_length]
        
        token_lists.append(audio_tokens)
    
    return token_lists