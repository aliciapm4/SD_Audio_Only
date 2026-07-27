"""
3_multi_factor_distance_engine.py
Calculates relative proximity metrics across all 4 body-worn microphones.
"""

import numpy as np
import librosa
from scipy.signal import correlate, correlation_lags, butter, filtfilt

class MultiFactorDistanceEngine:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
    def compute_rss(self, audio_slices):
        """
        Calculates Root-Mean-Square (RMS) energy ratios relative to the primary speaker's mic.
        """
        rms_energies = np.array([np.sqrt(np.mean(mic**2)) for mic in audio_slices])
        primary_idx = np.argmax(rms_energies)
        primary_energy = rms_energies[primary_idx]
        
        # Avoid division by zero
        rss_ratios = rms_energies / (primary_energy + 1e-9)
        return primary_idx, rss_ratios

    def compute_tdoa(self, audio_slices, primary_idx):
        """
        Calculates delay lags in milliseconds relative to the primary speaker track.
        """
        ref_signal = audio_slices[primary_idx]
        delays_ms = []
        
        for i, sig in enumerate(audio_slices):
            if i == primary_idx:
                delays_ms.append(0.0)
                continue
            
            corr = correlate(ref_signal, sig, mode='full')
            lags = correlation_lags(len(ref_signal), len(sig), mode='full')
            peak_lag = lags[np.argmax(corr)]
            
            delay_sec = abs(peak_lag) / self.sr
            delays_ms.append(round(delay_sec * 1000.0, 3))
            
        return np.array(delays_ms)

    def compute_hf_attenuation(self, audio_slices):
        """
        Measures ratio of high-frequency (>2 kHz) energy to total energy.
        Lower ratios indicate acoustic shadowing (turning away or greater distance).
        """
        b, a = butter(4, 2000 / (self.sr / 2), btype='highpass')
        hf_ratios = []
        
        for sig in audio_slices:
            hf_sig = filtfilt(b, a, sig)
            hf_energy = np.sum(hf_sig**2)
            total_energy = np.sum(sig**2) + 1e-9
            hf_ratios.append(round(float(hf_energy / total_energy), 4))
            
        return np.array(hf_ratios)

    def analyze_speech_segment(self, audio_slices):
        """
        Runs all three proximity metrics for a single speech frame slice.
        """
        primary_idx, rss_ratios = self.compute_rss(audio_slices)
        tdoa_delays = self.compute_tdoa(audio_slices, primary_idx)
        hf_ratios = self.compute_hf_attenuation(audio_slices)
        
        # Rank channels by proximity (highest volume bleed = closest)
        sorted_indices = np.argsort(-rss_ratios)
        proximity_order = [f"Mic {i+1}" for i in sorted_indices]
        
        return {
            'primary_speaker_mic': primary_idx + 1,
            'proximity_ranking': " -> ".join(proximity_order),
            'rss_ratios': np.round(rss_ratios, 4).tolist(),
            'tdoa_delays_ms': tdoa_delays.tolist(),
            'hf_ratios': hf_ratios.tolist()
        }

if __name__ == "__main__":
    pass