import csv
import re
from itertools import combinations
from typing import List, Dict
import pandas as pd

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
N_WORDS_MATCH = 4              # Min words to identify a phrase match

# Max allowed time window (in seconds) between Person A speaking and Person B responding
# to consider it a direct conversational turn-taking pair
MAX_RESPONSE_GAP_SEC = 30.0    

SPEED_OF_SOUND_MPS = 343.0     # Speed of sound (m/s)
SPEED_OF_SOUND_FPS = 1125.33   # Speed of sound (ft/s)

OUTPUT_CSV_FILE = "two_way_interaction_distances.csv"

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
    """Strips LENA confidence scores and punctuation."""
    cleaned = re.sub(r"\(\d+\.\d+\)", "", str(raw_text))
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned.strip().lower()


def load_ngrams_from_excel(file_path: str, n: int) -> List[Dict]:
    """Extracts sliding window n-grams with timestamps from an Excel transcript."""
    df = pd.read_excel(file_path)
    words_data = []

    for _, row in df.iterrows():
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
            words_data.append({
                "word": word,
                "timestamp": t_start + (idx * time_per_word)
            })

    ngrams = []
    for i in range(len(words_data) - n + 1):
        window = words_data[i: i + n]
        phrase = " ".join([w["word"] for w in window])
        ngrams.append({
            "phrase": phrase,
            "timestamp": window[0]["timestamp"]
        })

    return ngrams


def find_phrase_overlaps(ngrams_a: List[Dict], ngrams_b: List[Dict]) -> List[Dict]:
    """
    Finds all instances of identical phrases captured on both Mic A and Mic B.
    Returns delta_t = Time_B - Time_A for each match.
    """
    map_b = {}
    for ng in ngrams_b:
        map_b.setdefault(ng["phrase"], []).append(ng["timestamp"])

    overlaps = []
    for ng_a in ngrams_a:
        phrase = ng_a["phrase"]
        if phrase in map_b:
            for t_b in map_b[phrase]:
                t_a = ng_a["timestamp"]
                overlaps.append({
                    "phrase": phrase,
                    "time_a": t_a,
                    "time_b": t_b,
                    "delta_t_b_minus_a": t_b - t_a
                })
    return overlaps


def run_two_way_interaction_analysis():
    print("Loading transcripts...")
    transcripts_ngrams = {}

    for speaker, path in TRANSCRIPT_FILES.items():
        try:
            transcripts_ngrams[speaker] = load_ngrams_from_excel(path, N_WORDS_MATCH)
            print(f" - Loaded {speaker}: {len(transcripts_ngrams[speaker])} n-grams.")
        except Exception as e:
            print(f" ⚠️ Could not load file for {speaker}: {e}")
            return

    results = []
    speaker_pairs = list(combinations(TRANSCRIPT_FILES.keys(), 2))

    print("\n--- ANALYZING TWO-WAY CONVERSATIONAL EXCHANGES ---")

    for person_a, person_b in speaker_pairs:
        ngrams_a = transcripts_ngrams[person_a]
        ngrams_b = transcripts_ngrams[person_b]

        # Step 1: Find all shared audio occurrences between Mic A and Mic B
        overlaps = find_phrase_overlaps(ngrams_a, ngrams_b)

        if not overlaps:
            continue

        # Step 2: Search for reciprocal 2-way exchanges (A -> B followed by B -> A)
        for i in range(len(overlaps)):
            match1 = overlaps[i]
            # Match 1: Direction A -> B
            delta_1 = match1["delta_t_b_minus_a"]

            for j in range(i + 1, len(overlaps)):
                match2 = overlaps[j]

                # Check if Match 2 occurs shortly after Match 1 within response window
                time_gap = match2["time_a"] - match1["time_a"]
                if 0 < time_gap <= MAX_RESPONSE_GAP_SEC:
                    delta_2 = match2["delta_t_b_minus_a"]

                    # Calculate pure acoustic propagation delay tau
                    # tau = |delta_1 + (-delta_2)| / 2 = |delta_1 - delta_2| / 2
                    # Depending on who spoke first/second
                    tau = abs(delta_1 - delta_2) / 2.0
                    
                    # Calculate local clock offset delta_O
                    clock_offset = (delta_1 + delta_2) / 2.0

                    dist_m = tau * SPEED_OF_SOUND_MPS
                    dist_ft = tau * SPEED_OF_SOUND_FPS

                    results.append({
                        "Speaker_A": person_a,
                        "Speaker_B": person_b,
                        "Phrase_1": match1["phrase"],
                        "Time_A_Phrase_1": round(match1["time_a"], 3),
                        "Time_B_Phrase_1": round(match1["time_b"], 3),
                        "Phrase_2": match2["phrase"],
                        "Time_A_Phrase_2": round(match2["time_a"], 3),
                        "Time_B_Phrase_2": round(match2["time_b"], 3),
                        "Calculated_Acoustic_Delay_Sec": round(tau, 4),
                        "Estimated_Clock_Offset_Sec": round(clock_offset, 3),
                        "Distance_Feet": round(dist_ft, 2),
                        "Distance_Meters": round(dist_m, 2)
                    })

    # Sort results chronologically
    results.sort(key=lambda x: (x["Speaker_A"], x["Speaker_B"], x["Time_A_Phrase_1"]))

    # Step 3: Write out to CSV
    fieldnames = [
        "Speaker_A",
        "Speaker_B",
        "Phrase_1",
        "Time_A_Phrase_1",
        "Time_B_Phrase_1",
        "Phrase_2",
        "Time_A_Phrase_2",
        "Time_B_Phrase_2",
        "Calculated_Acoustic_Delay_Sec",
        "Estimated_Clock_Offset_Sec",
        "Distance_Feet",
        "Distance_Meters"
    ]

    with open(OUTPUT_CSV_FILE, mode="w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAnalysis complete! Extracted {len(results)} two-way exchange instances.")
    print(f"Results saved to: '{OUTPUT_CSV_FILE}'")


if __name__ == "__main__":
    run_two_way_interaction_analysis()