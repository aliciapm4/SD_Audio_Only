import csv
import re
from itertools import combinations
from typing import List, Dict
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
# Minimum consecutive words required for an exact audio match
N_WORDS_MATCH = 4

# Maximum allowed search window (in seconds) to consider two matching phrases part of the same trial event.
# Set generously (e.g. 120s) so it catches trial start drifts across concatenated chunks.
MAX_SEARCH_WINDOW_SEC = 120.0 

# Speed of sound for rough distance estimation
SPEED_OF_SOUND_MPS = 343.0      # m/s
SPEED_OF_SOUND_FPS = 1125.33    # ft/s

# Speaker used as the anchor clock reference
REFERENCE_SPEAKER = "Max"

OUTPUT_CSV_FILE = "piecewise_aligned_matches.csv"

TRANSCRIPT_FILES = {
    "Max": "LENA_UNIT4JH_MAX_06262026.xlsx",
    "Riad": "LENA_UNIT5JH_RIAD_06262026.xlsx",
    "Kush": "LENA_UNIT6_KUSH_06262026.xlsx",
    "Alicia": "LENA_UNIT8_ALICIA_06262026.xlsx"
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_timestamp(timestamp_val) -> float:
    """Converts HH:MM:SS.mmm to seconds."""
    timestamp_str = str(timestamp_val).strip()
    try:
        parts = timestamp_str.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return 0.0


def clean_text(raw_text: str) -> str:
    """Removes LENA confidence scores like (0.8459) and punctuation."""
    cleaned = re.sub(r"\(\d+\.\d+\)", "", str(raw_text))
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned.strip().lower()


def load_word_stream(file_path: str) -> List[Dict]:
    """Reads Excel file and constructs a continuous tokenized word stream with estimated timestamps."""
    df = pd.read_excel(file_path)
    words_stream = []

    for row_idx, row in df.iterrows():
        start_val = row.get("Segment Start Time", "")
        end_val = row.get("Segment End Time", "")
        raw_text = row.get("Transcription (Confidence)", "")

        if pd.isna(start_val) or pd.isna(raw_text):
            continue

        t_start = parse_timestamp(start_val)
        t_end = parse_timestamp(end_val)
        cleaned = clean_text(raw_text)
        words = cleaned.split()

        if not words:
            continue

        duration = max(t_end - t_start, 0.1)
        time_per_word = duration / len(words)

        for idx, word in enumerate(words):
            words_stream.append({
                "word": word,
                "timestamp": t_start + (idx * time_per_word),
                "raw_segment_text": str(raw_text),
                "row_idx": row_idx
            })

    return words_stream


def extract_ngrams(words_stream: List[Dict], n: int) -> List[Dict]:
    """Generates sliding window n-grams across the continuous stream."""
    ngrams = []
    for i in range(len(words_stream) - n + 1):
        window = words_stream[i : i + n]
        phrase = " ".join([w["word"] for w in window])
        start_time = window[0]["timestamp"]

        ngrams.append({
            "phrase": phrase,
            "start_time": start_time,
            "raw_text": window[0]["raw_segment_text"]
        })
    return ngrams


def estimate_local_offsets(transcripts_ngrams: Dict[str, List[Dict]], ref_speaker: str, search_window: float) -> Dict[str, List[Dict]]:
    """
    Computes piece-wise local clock offsets across time.
    For each matched phrase in the recording, it measures the local offset relative to the reference speaker.
    """
    ref_ngrams = transcripts_ngrams[ref_speaker]
    
    # Map reference speaker phrases to their timestamps
    ref_map = {}
    for ng in ref_ngrams:
        ref_map.setdefault(ng["phrase"], []).append(ng["start_time"])

    speaker_offsets = {}

    for speaker, ngrams in transcripts_ngrams.items():
        if speaker == ref_speaker:
            continue

        local_matches = []
        for ng in ngrams:
            phrase = ng["phrase"]
            t_target = ng["start_time"]

            if phrase in ref_map:
                # Find closest corresponding phrase match in the reference timeline
                best_ref_time = min(ref_map[phrase], key=lambda t_ref: abs(t_ref - t_target))
                local_offset = t_target - best_ref_time

                if abs(local_offset) <= search_window:
                    local_matches.append({
                        "target_time": t_target,
                        "ref_time": best_ref_time,
                        "local_offset": local_offset,
                        "phrase": phrase
                    })

        speaker_offsets[speaker] = local_matches

    return speaker_offsets


def run_piecewise_overlap_analysis():
    """Executes multi-trial piecewise alignment and outputs distance estimates."""
    print("Loading Excel transcript files...")
    transcripts_ngrams = {}
    
    for speaker, path in TRANSCRIPT_FILES.items():
        try:
            words = load_word_stream(path)
            ngrams = extract_ngrams(words, N_WORDS_MATCH)
            transcripts_ngrams[speaker] = ngrams
            print(f" - Loaded {speaker}: {len(words)} words, {len(ngrams)} phrase blocks.")
        except Exception as e:
            print(f" ⚠️ Could not load file for {speaker}: {e}")
            return

    # Step 1: Compute local offsets for each speaker relative to Reference (Max)
    print(f"\nComputing local time offsets relative to Reference Speaker: [{REFERENCE_SPEAKER}]...")
    speaker_local_offsets = estimate_local_offsets(
        transcripts_ngrams, 
        ref_speaker=REFERENCE_SPEAKER, 
        search_window=MAX_SEARCH_WINDOW_SEC
    )

    results = []
    person_pairs = list(combinations(TRANSCRIPT_FILES.keys(), 2))

    # Step 2: Compare pairs using piecewise alignment
    for person_a, person_b in person_pairs:
        ngrams_a = transcripts_ngrams[person_a]
        ngrams_b = transcripts_ngrams[person_b]

        map_b = {}
        for ng in ngrams_b:
            map_b.setdefault(ng["phrase"], []).append(ng)

        for ng_a in ngrams_a:
            phrase = ng_a["phrase"]
            if phrase in map_b:
                for ng_b in map_b[phrase]:
                    raw_diff = abs(ng_a["start_time"] - ng_b["start_time"])

                    # Accept match if it occurs within max search window across concatenated segments
                    if raw_diff <= MAX_SEARCH_WINDOW_SEC:
                        
                        # Distance estimation based on residual time difference
                        # (Note: transcript resolution is ~1s, so this yields a coarse macro-estimate)
                        dist_m = raw_diff * SPEED_OF_SOUND_MPS
                        dist_ft = raw_diff * SPEED_OF_SOUND_FPS

                        results.append({
                            "Speaker_1": person_a,
                            "Speaker_2": person_b,
                            "Speaker_1_Time_Sec": round(ng_a["start_time"], 3),
                            "Speaker_2_Time_Sec": round(ng_b["start_time"], 3),
                            "Raw_Time_Diff_Sec": round(raw_diff, 4),
                            "Matched_Phrase": phrase,
                            "Estimated_Distance_Ft": round(dist_ft, 2),
                            "Estimated_Distance_M": round(dist_m, 2)
                        })

    # Deduplicate closely adjacent sliding window matches
    deduped_results = []
    seen = set()
    for res in sorted(results, key=lambda x: (x["Speaker_1"], x["Speaker_1_Time_Sec"])):
        key = (res["Speaker_1"], res["Speaker_2"], round(res["Speaker_1_Time_Sec"], 1), res["Matched_Phrase"])
        if key not in seen:
            seen.add(key)
            deduped_results.append(res)

    # Step 3: Export results to CSV
    fieldnames = [
        "Speaker_1",
        "Speaker_2",
        "Speaker_1_Time_Sec",
        "Speaker_2_Time_Sec",
        "Raw_Time_Diff_Sec",
        "Matched_Phrase",
        "Estimated_Distance_Ft",
        "Estimated_Distance_M"
    ]

    with open(OUTPUT_CSV_FILE, mode="w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_results)

    print(f"\nProcessing complete! Found {len(deduped_results)} matched phrase instances across all concatenated trials.")
    print(f"Results exported to: '{OUTPUT_CSV_FILE}'")


if __name__ == "__main__":
    run_piecewise_overlap_analysis()