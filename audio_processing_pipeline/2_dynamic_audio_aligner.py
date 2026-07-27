"""
2_dynamic_audio_aligner.py
Performs sub-millisecond audio alignment per 50-second movement segment
to compensate for LENA unit internal clock drift.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate, correlation_lags, butter, filtfilt

def align_audio_window(audio_signals, sr=16000, reference_idx=0):
    """
    Aligns a multi-channel audio slice relative to a reference channel using cross-correlation.
    """
    ref_sig = audio_signals[reference_idx]
    aligned_signals = []
    
    # Apply bandpass filter around 2 kHz to enhance tone/speech transients
    b, a = butter(4, [1800 / (sr / 2), 2200 / (sr / 2)], btype='bandpass')
    ref_filtered = filtfilt(b, a, ref_sig)
    
    offsets = []
    for i, sig in enumerate(audio_signals):
        if i == reference_idx:
            offsets.append(0)
            aligned_signals.append(sig)
            continue
            
        sig_filtered = filtfilt(b, a, sig)
        corr = correlate(ref_filtered, sig_filtered, mode='full')
        lags = correlation_lags(len(ref_filtered), len(sig_filtered), mode='full')
        
        peak_lag = lags[np.argmax(corr)]
        offsets.append(peak_lag)
        
        # Shift signal
        if peak_lag > 0:
            shifted = np.pad(sig, (peak_lag, 0), mode='constant')[:len(sig)]
        elif peak_lag < 0:
            shifted = np.pad(sig[abs(peak_lag):], (0, abs(peak_lag)), mode='constant')
        else:
            shifted = sig
            
        aligned_signals.append(shifted)
        
    return np.array(aligned_signals), offsets

def process_and_export_aligned_trial(file_paths, output_paths, sr=16000, segment_duration=50.0):
    """
    Splits audio into 50-second movement blocks, aligns each block independently to eliminate
    accumulated clock drift, and saves the corrected synchronized audio files.
    """
    signals = [librosa.load(p, sr=sr, mono=True)[0] for p in file_paths]
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]
    
    samples_per_block = int(segment_duration * sr)
    num_blocks = min_len // samples_per_block
    
    corrected_channels = [[] for _ in file_paths]
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * samples_per_block
        end_idx = start_idx + samples_per_block
        
        block_slices = [s[start_idx:end_idx] for s in signals]
        aligned_block, offsets = align_audio_window(block_slices, sr=sr, reference_idx=0)
        
        for ch in range(len(file_paths)):
            corrected_channels[ch].extend(aligned_block[ch])
            
    # Export fully synchronized tracks
    for path, sig in zip(output_paths, corrected_channels):
        sf.write(path, np.array(sig), sr)
    print("Dynamically aligned audio exported successfully.")

if __name__ == "__main__":
    pass