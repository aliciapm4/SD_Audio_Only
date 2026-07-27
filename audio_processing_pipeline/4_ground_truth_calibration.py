"""
4_ground_truth_calibrator.py
Fits an empirical power-law model mapping audio RSS energy ratios to distance in feet.
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def power_law(rss_ratio, a, b):
    """
    Inverse-Square Law derivative model: Distance = a * (RSS_Ratio ^ b)
    """
    return a * (rss_ratio ** b)

class GroundTruthCalibrator:
    def __init__(self):
        self.a = None
        self.b = None
        
    def fit(self, observed_rss_ratios, ground_truth_feet):
        """
        Fits empirical observations against ground-truth grid distances (2ft, 6ft, 12ft).
        """
        rss_arr = np.array(observed_rss_ratios, dtype=float)
        dist_arr = np.array(ground_truth_feet, dtype=float)
        
        popt, _ = curve_fit(power_law, rss_arr, dist_arr, p0=[1.0, -0.5], maxfev=10000)
        self.a, self.b = popt
        
        preds = power_law(rss_arr, self.a, self.b)
        r2 = r2_score(dist_arr, preds)
        
        print("=== GROUND-TRUTH CALIBRATION COMPLETE ===")
        print(f"Model: Distance (ft) = {self.a:.4f} * (RSS_Ratio ^ {self.b:.4f})")
        print(f"R² Score: {r2:.4f}\n")
        return self.a, self.b

    def predict_feet(self, rss_ratio):
        """
        Converts a single RSS bleed ratio into distance in feet.
        """
        if self.a is None or self.b is None:
            raise ValueError("Model must be calibrated before calling predict_feet().")
            
        if rss_ratio >= 1.0:
            return 0.0
        if rss_ratio <= 0.0:
            return float('inf')
            
        return round(float(power_law(rss_ratio, self.a, self.b)), 2)

if __name__ == "__main__":
    # Example calibration test using 2ft, 6ft, 12ft grid ground-truth values
    sample_rss = [0.85, 0.82, 0.32, 0.29, 0.09, 0.07]
    sample_feet = [2.0, 2.0, 6.0, 6.0, 12.0, 12.0]
    
    calibrator = GroundTruthCalibrator()
    calibrator.fit(sample_rss, sample_feet)
    print(f"Predicted distance for 0.40 RSS bleed: {calibrator.predict_feet(0.40)} ft")