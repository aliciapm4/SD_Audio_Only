"""
main_pipeline.py
Master execution pipeline combining transcript slicing, dynamic alignment,
audio extraction, metric evaluation, and calibrated distance predictions.
"""

import os
import numpy as np
import librosa

from transcript_slicer import crop_master_transcript, extract_solo_speech_segments
from dynamic_audio_aligner import process_and_export_aligned_trial
from multi_factor_distance_engine import MultiFactorDistanceEngine
from ground_truth_calibrator import GroundTruthCalibrator

def run_pipeline():
    # 1. File Configuration
    raw_audio_files = [
        "raw_waveforms/p1_trial1.wav",
        "raw_waveforms/p2_trial1.wav",
        "raw_waveforms/p3_trial1.wav",
        "raw_waveforms/p4_trial1.wav"
    ]
    
    aligned_audio_files = [
        "aligned_waveforms/p1_trial1_synced.wav",
        "aligned_waveforms/p2_trial1_synced.wav",
        "aligned_waveforms/p3_trial1_synced.wav",
        "aligned_waveforms/p4_trial1_synced.wav"
    ]
    
    os.makedirs("aligned_waveforms", exist_ok=True)

    # 2. Dynamic Clock-Drift Correction and Alignment
    print("Step 1: Dynamically aligning audio files to eliminate clock drift...")
    process_and_export_aligned_trial(raw_audio_files, aligned_audio_files, sr=16000)

    # 3. Load Synced Audio Files
    signals = [librosa.load(p, sr=16000, mono=True)[0] for p in aligned_audio_files]

    # 4. Calibration with Ground Truth Grid Data (Trial 1=2ft, Trial 2=6ft, Trial 3=12ft)
    print("\nStep 2: Calibrating model using Ground Truth Grid trials...")
    training_rss_observations = [0.84, 0.81, 0.33, 0.30, 0.09, 0.07]
    training_grid_feet = [2.0, 2.0, 6.0, 6.0, 12.0, 12.0]
    
    calibrator = GroundTruthCalibrator()
    calibrator.fit(training_rss_observations, training_grid_feet)

    # 5. Extract Solo Speech Windows from Transcript
    print("Step 3: Extracting non-overlapping speech windows...")
    sample_speaker_transcripts = {
        'p1': [{'start': 12.0, 'end': 15.5}, {'start': 65.0, 'end': 68.0}],
        'p2': [{'start': 22.0, 'end': 25.0}],
        'p3': [{'start': 80.0, 'end': 84.0}],
        'p4': [{'start': 110.0, 'end': 114.0}]
    }
    
    solo_windows = extract_solo_speech_segments(sample_speaker_transcripts)
    
    # 6. Execute Distance Analysis
    print("\nStep 4: Running Multi-Factor Spatial Distance Analysis...")
    engine = MultiFactorDistanceEngine(sample_rate=16000)
    
    for window in solo_windows:
        start_frame = int(window['start'] * 16000)
        end_frame = int(window['end'] * 16000)
        
        audio_slices = [sig[start_frame:end_frame] for sig in signals]
        metrics = engine.analyze_speech_segment(audio_slices)
        
        print(f"[{window['start']}s - {window['end']}s] Speaker {window['speaker']} (Mic {metrics['primary_speaker_mic']}):")
        print(f"   -> Proximity Ranking: {metrics['proximity_ranking']}")
        
        # Predict physical distance in feet for secondary channels
        for ch_idx, rss_ratio in enumerate(metrics['rss_ratios']):
            dist_ft = calibrator.predict_feet(rss_ratio)
            print(f"      - Mic {ch_idx+1}: {rss_ratio:.3f} RSS Ratio -> Estimated {dist_ft} ft")
        print()

if __name__ == "__main__":
    run_pipeline()