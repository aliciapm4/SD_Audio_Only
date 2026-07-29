import csv
import re
from itertools import combinations
from typing import List, Dict
import pandas as pd

# ==========================================
# CONFIGURABLE PARAMETERS
# ==========================================
N_WORDS_MATCH = 4            # Min consecutive words to match
MAX_TIME_DIFF_SEC = 20.0     # Max allowed time difference in seconds

SPEED_OF_SOUND_MPS = 343.0   # Speed of sound (m/s)
SPEED_OF_SOUND_FPS = 1125.33 # Speed of sound (ft/s)

OUTPUT_CSV_FILE = "speaker_proximity_results.csv"

# Update these filenames to match your exact .xlsx file paths
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
    """Converts HH:MM:SS.mmm string or timestamp object into float seconds."""
    timestamp_str = str(timestamp_val).strip()
    try:
        parts = timestamp_str.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return 0.0


def clean_transcription_text(raw_text: str) -> str:
    """Strips confidence scores like '(0.8459)', punctuation, and lowercases text."""
    cleaned = re.sub(r"\(\d+\.\d+\)", "", str(raw_text))
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned.strip().lower()


def load_transcript_words_from_excel(file_path: str) -> List[Dict]:
    """Reads Excel file using pandas and extracts words with estimated timestamps."""
    words_data = []

    # Read the Excel file
    df = pd.read_excel(file_path)

    for _, row in df.iterrows():
        start_val = row.get("Segment Start Time", "")
        end_val = row.get("Segment End Time", "")
        raw_text = row.get("Transcription (Confidence)", "")

        if pd.isna(start_val) or pd.isna(raw_text):
            continue

        t_start = parse_timestamp(start_val)
        t_end = parse_timestamp(end_val)
        cleaned_text = clean_transcription_text(raw_text)

        words = cleaned_text.split()
        if not words:
            continue

        duration = max(t_end - t_start, 0.1)
        time_per_word = duration / len(words)

        for idx, word in enumerate(words):
            word_time = t_start + (idx * time_per_word)
            words_data.append({
                "word": word,
                "timestamp": word_time
            })

    return words_data


def extract_ngrams(words_data: List[Dict], n: int) -> List[Dict]:
    """Extracts sliding window n-grams with timestamps."""
    ngrams = []
    for i in range(len(words_data) - n + 1):
        window = words_data[i: i + n]
        phrase = " ".join([w["word"] for w in window])
        start_time = window[0]["timestamp"]

        ngrams.append({
            "phrase": phrase,
            "start_time": start_time
        })
    return ngrams


def find_and_export_overlaps(file_dict: Dict[str, str], n_words: int, max_time_diff: float, output_csv: str):
    """Matches overlapping phrases across speaker pairs and outputs results to a CSV."""
    transcripts_data = {}
    print("Loading Excel transcripts...")
    for person, path in file_dict.items():
        try:
            words = load_transcript_words_from_excel(path)
            ngrams = extract_ngrams(words, n_words)
            transcripts_data[person] = ngrams
            print(f" - Loaded {person}: {len(words)} words, {len(ngrams)} phrase blocks.")
        except Exception as e:
            print(f" Warning: Failed to load '{path}' for {person}. Error: {e}")

    results = []
    person_pairs = list(combinations(transcripts_data.keys(), 2))

    for person_a, person_b in person_pairs:
        ngrams_a = transcripts_data[person_a]
        ngrams_b = transcripts_data[person_b]

        phrase_map_b: Dict[str, List[Dict]] = {}
        for ng_b in ngrams_b:
            phrase_map_b.setdefault(ng_b["phrase"], []).append(ng_b)

        for ng_a in ngrams_a:
            phrase = ng_a["phrase"]
            if phrase in phrase_map_b:
                for ng_b in phrase_map_b[phrase]:
                    time_diff = abs(ng_a["start_time"] - ng_b["start_time"])

                    if time_diff <= max_time_diff:
                        dist_m = time_diff * SPEED_OF_SOUND_MPS
                        dist_ft = time_diff * SPEED_OF_SOUND_FPS

                        results.append({
                            "Speaker_1": person_a,
                            "Speaker_2": person_b,
                            "Speaker_1_Time_Sec": round(ng_a["start_time"], 3),
                            "Speaker_2_Time_Sec": round(ng_b["start_time"], 3),
                            "Matched_Phrase": phrase,
                            "Time_Delta_Sec": round(time_diff, 4),
                            "Distance_Feet": round(dist_ft, 2),
                            "Distance_Meters": round(dist_m, 2)
                        })

    # Sort results chronologically
    results.sort(key=lambda x: (x["Speaker_1"], x["Speaker_1_Time_Sec"]))

    fieldnames = [
        "Speaker_1", 
        "Speaker_2", 
        "Speaker_1_Time_Sec", 
        "Speaker_2_Time_Sec", 
        "Matched_Phrase", 
        "Time_Delta_Sec", 
        "Distance_Feet", 
        "Distance_Meters"
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nProcessing complete! Exported {len(results)} overlap instances to '{output_csv}'.")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    find_and_export_overlaps(
        file_dict=TRANSCRIPT_FILES,
        n_words=N_WORDS_MATCH,
        max_time_diff=MAX_TIME_DIFF_SEC,
        output_csv=OUTPUT_CSV_FILE
    )