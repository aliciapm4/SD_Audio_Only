"""
1_transcript_slicer.py
Handles slicing master transcripts into trial-specific timeframes and
extracting solo-speaker speech windows to avoid cross-talk contamination.
"""

import json

def crop_master_transcript(master_transcript, trial_start_sec, trial_duration_sec=300.0):
    """
    Crops master transcript timestamps relative to a specific trial's start time (T0).
    """
    trial_end_sec = trial_start_sec + trial_duration_sec
    cropped_segments = []
    
    for seg in master_transcript:
        # Check if segment falls within trial duration
        if seg['start'] >= trial_start_sec and seg['end'] <= trial_end_sec:
            cropped_segments.append({
                'speaker': seg['speaker'],
                'text': seg.get('text', ''),
                'start': round(seg['start'] - trial_start_sec, 3),
                'end': round(seg['end'] - trial_start_sec, 3)
            })
            
    return cropped_segments

def extract_solo_speech_segments(speaker_transcripts, min_duration=1.5):
    """
    Scans transcripts across all 4 speakers and isolates time windows where 
    ONLY ONE person is speaking continuously for at least min_duration seconds.
    
    speaker_transcripts: dict {'p1': [segs], 'p2': [segs], 'p3': [segs], 'p4': [segs]}
    """
    all_segments = []
    for spk_id, segments in speaker_transcripts.items():
        for seg in segments:
            all_segments.append({
                'speaker': spk_id,
                'start': seg['start'],
                'end': seg['end']
            })
            
    all_segments.sort(key=lambda x: x['start'])
    solo_segments = []
    
    for i, current in enumerate(all_segments):
        duration = current['end'] - current['start']
        if duration < min_duration:
            continue
            
        has_overlap = False
        for j, other in enumerate(all_segments):
            if i == j or current['speaker'] == other['speaker']:
                continue
            # Overlap check: (StartA < EndB) and (EndA > StartB)
            if (current['start'] < other['end']) and (current['end'] > other['start']):
                has_overlap = True
                break
                
        if not has_overlap:
            solo_segments.append(current)
            
    return solo_segments

if __name__ == "__main__":
    # Example test
    sample_master = [
        {'speaker': 'p1', 'start': 105.0, 'end': 109.2, 'text': 'Hello world'},
        {'speaker': 'p2', 'start': 108.0, 'end': 111.0, 'text': 'Overlap speech'},
        {'speaker': 'p3', 'start': 120.0, 'end': 124.5, 'text': 'Solo speech here'}
    ]
    
    # Crop trial starting at T0 = 100.0s
    trial1_transcripts = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
    for seg in crop_master_transcript(sample_master, trial_start_sec=100.0, trial_duration_sec=300.0):
        trial1_transcripts[seg['speaker']].append(seg)
        
    solos = extract_solo_speech_segments(trial1_transcripts)
    print(f"Extracted {len(solos)} solo speech windows.")