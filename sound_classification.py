"""
EcoSonic AI - Sound Classification Module

DEMO MODE: This module uses feature-based audio analysis as a demonstration
classifier. It is NOT a trained deep learning model. Results are approximate.

To connect a trained model, replace the predict() method in DemoClassifier.
"""
import numpy as np
from typing import Dict, Tuple

DEMO_MODE = True
MODEL_VERSION = "demo-v1"


class DemoClassifier:
    """
    Feature-based demo classifier for bioacoustic events.
    Uses spectral centroid, ZCR, bandwidth, MFCCs to estimate categories.
    """
    CATEGORIES = ["Bird Calls", "Frog Calls", "Insect Sounds", "Unknown/Other"]

    def __init__(self):
        self.model_version = MODEL_VERSION
        self.is_demo = True
        self.feature_profiles = {
            "Bird Calls": {
                "spectral_centroid_range": (1500, 6000),
                "zcr_range": (0.02, 0.15),
                "bandwidth_range": (1500, 5000),
                "mfcc1_range": (-300, -50),
                "spectral_contrast_weight": 1.2,
            },
            "Frog Calls": {
                "spectral_centroid_range": (500, 3500),
                "zcr_range": (0.01, 0.08),
                "bandwidth_range": (500, 2500),
                "mfcc1_range": (-400, -100),
                "spectral_contrast_weight": 0.8,
            },
            "Insect Sounds": {
                "spectral_centroid_range": (3000, 10000),
                "zcr_range": (0.08, 0.35),
                "bandwidth_range": (800, 3000),
                "mfcc1_range": (-250, 0),
                "spectral_contrast_weight": 0.5,
            },
        }

    def _score_category(self, features: Dict, category: str) -> float:
        profile = self.feature_profiles[category]
        score = 0.0
        total_weight = 0.0

        centroid = features.get("spectral_centroid_mean", 2000)
        c_low, c_high = profile["spectral_centroid_range"]
        c_mid = (c_low + c_high) / 2
        c_range = (c_high - c_low) / 2
        centroid_score = max(0, 1.0 - abs(centroid - c_mid) / (c_range * 2)) if c_range > 0 else 0.5
        score += centroid_score * 3.0
        total_weight += 3.0

        zcr = features.get("zero_crossing_rate_mean", 0.05)
        z_low, z_high = profile["zcr_range"]
        z_mid = (z_low + z_high) / 2
        z_range = (z_high - z_low) / 2
        zcr_score = max(0, 1.0 - abs(zcr - z_mid) / (z_range * 2)) if z_range > 0 else 0.5
        score += zcr_score * 2.0
        total_weight += 2.0

        bw = features.get("spectral_bandwidth_mean", 2000)
        b_low, b_high = profile["bandwidth_range"]
        b_mid = (b_low + b_high) / 2
        b_range = (b_high - b_low) / 2
        bw_score = max(0, 1.0 - abs(bw - b_mid) / (b_range * 2)) if b_range > 0 else 0.5
        score += bw_score * 2.0
        total_weight += 2.0

        mfcc1 = features.get("mfcc_1_mean", -200)
        m_low, m_high = profile["mfcc1_range"]
        m_mid = (m_low + m_high) / 2
        m_range = (m_high - m_low) / 2
        mfcc_score = max(0, 1.0 - abs(mfcc1 - m_mid) / (m_range * 2)) if m_range > 0 else 0.5
        score += mfcc_score * 2.0
        total_weight += 2.0

        contrast = features.get("spectral_contrast_mean", 20)
        contrast_norm = min(1.0, contrast / 40.0)
        sc_weight = profile["spectral_contrast_weight"]
        score += contrast_norm * sc_weight
        total_weight += sc_weight

        return score / total_weight if total_weight > 0 else 0.25

    def predict(self, features: Dict) -> Tuple[str, float, Dict[str, float]]:
        scores = {}
        for category in ["Bird Calls", "Frog Calls", "Insect Sounds"]:
            scores[category] = self._score_category(features, category)

        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] = (scores[k] / total) * 0.88

        max_score = max(scores.values()) if scores else 0
        unknown_score = max(0.04, 1.0 - sum(scores.values()))

        if max_score < 0.35:
            unknown_score = max(unknown_score, 0.3)
            remaining = 1.0 - unknown_score
            s_total = sum(scores.values())
            for k in scores:
                scores[k] = (scores[k] / s_total) * remaining if s_total > 0 else remaining / 3

        scores["Unknown/Other"] = round(unknown_score, 4)
        for k in scores:
            scores[k] = round(scores[k], 4)

        total = sum(scores.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            for k in scores:
                scores[k] = round(scores[k] / total, 4)

        predicted = max(scores, key=scores.get)
        confidence = scores[predicted]
        return predicted, confidence, scores


def get_classifier():
    if DEMO_MODE:
        return DemoClassifier()
    else:
        raise NotImplementedError("Trained model not available. Set DEMO_MODE = True.")


def classify_audio(features: Dict) -> Dict:
    """Main classification function."""
    classifier = get_classifier()
    predicted_category, confidence, all_scores = classifier.predict(features)
    result = {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "all_scores": all_scores,
        "model_version": classifier.model_version,
        "is_demo": classifier.is_demo,
        "demo_notice": (
            "DEMO MODE: This classification uses feature-based audio analysis, "
            "not a trained deep learning model. Results are approximate."
        ) if classifier.is_demo else None,
    }
    if confidence >= 0.75:
        result["confidence_level"] = "High"
    elif confidence >= 0.40:
        result["confidence_level"] = "Moderate"
    else:
        result["confidence_level"] = "Low"
        result["low_confidence_warning"] = (
            "Low confidence prediction. The audio may contain multiple overlapping "
            "sounds, background noise, or unrepresented sound types. Expert review recommended."
        )
    return result
