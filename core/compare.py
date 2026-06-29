from typing import Any, Dict
from lib.scene_detector import compare_scene_structures
from lib.audio import compare_audio_fingerprints
from lib.motion_hasher import compare_motion_fingerprints
from lib.phash import compare_video_fingerprints_with_timeline
from lib.blake3 import blake3_compare_fingerprint
from lib.ai_embedding import compare_ai_fingerprints
from lib.config import config

def compare_video(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares two videos sequentially based on active fingerprinting algorithms in config.yml.
    If similarity reaches or exceeds the threshold configured for a method,
    the comparison exits early and returns the duplicate result.
    """
    
    # 1. Compare Blake3 (100% binary matching on files)
    if config["features"]["blake3"]["enabled"]:
        blake_match = blake3_compare_fingerprint(v1.get('blake', ''), v2.get('blake', ''))
        if blake_match:
            return {
                "status": "duplicate",
                "match_type": "blake3",
                "match_percentage": 100.0,
                "message": "Both video files are binary identical (100% match on file hash).",
                "matched_segments": []
            }

    # 2. Compare pHash (Per-Frame Visual Matching)
    phash_pct = 0.0
    phash_res = {}
    if config["features"]["phash"]["enabled"]:
        phash_res = compare_video_fingerprints_with_timeline(v1.get('phash', []), v2.get('phash', []))
        phash_pct = phash_res.get("match_percentage", 0.0)
        dup_pct = config["features"]["phash"]["duplicate_threshold_pct"]
        if phash_pct >= dup_pct:
            return {
                "status": "duplicate",
                "match_type": "phash",
                "match_percentage": phash_pct,
                "message": f"High visual pHash similarity ({phash_pct}% >= threshold {dup_pct}%).",
                "matched_segments": phash_res.get("matched_segments", [])
            }

    # 3. Compare AI Visual Embeddings (Semantic Visual Matching)
    ai_pct = 0.0
    ai_res = {}
    if config["features"]["ai_embedding"]["enabled"]:
        ai_res = compare_ai_fingerprints(
            v1.get('ai', {}).get('ai_embeddings', []),
            v2.get('ai', {}).get('ai_embeddings', [])
        )
        ai_pct = ai_res.get("match_percentage", 0.0)
        dup_pct = config["features"]["ai_embedding"]["duplicate_threshold_pct"]
        if ai_pct >= dup_pct:
            return {
                "status": "duplicate",
                "match_type": "ai_embedding",
                "match_percentage": ai_pct,
                "message": f"High semantic visual similarity (AI) ({ai_pct}% >= threshold {dup_pct}%).",
                "matched_segments": ai_res.get("matched_segments", [])
            }

    # 4. Compare Audio Chromaprint (Voice/Audio Matching)
    audio_pct = 0.0
    audio_res = {}
    if config["features"]["audio"]["enabled"]:
        audio_res = compare_audio_fingerprints(
            v1.get('audio', {}).get('audio_subprints', []),
            v2.get('audio', {}).get('audio_subprints', [])
        )
        audio_pct = audio_res.get("match_percentage", 0.0)
        dup_pct = config["features"]["audio"]["duplicate_threshold_pct"]
        if audio_pct >= dup_pct:
            return {
                "status": "duplicate",
                "match_type": "audio",
                "match_percentage": audio_pct,
                "message": f"High audio track similarity ({audio_pct}% >= threshold {dup_pct}%).",
                "matched_segments": audio_res.get("matched_segments", [])
            }

    # 5. Compare Motion (Optical Flow Movement Pattern Matching)
    motion_pct = 0.0
    motion_res = {}
    if config["features"]["motion"]["enabled"]:
        motion_res = compare_motion_fingerprints(
            v1.get('motion', {}).get('motion_fingerprints', []),
            v2.get('motion', {}).get('motion_fingerprints', [])
        )
        motion_pct = motion_res.get("match_percentage", 0.0)
        dup_pct = config["features"]["motion"]["duplicate_threshold_pct"]
        if motion_pct >= dup_pct:
            return {
                "status": "duplicate",
                "match_type": "motion",
                "match_percentage": motion_pct,
                "message": f"High video motion pattern similarity ({motion_pct}% >= threshold {dup_pct}%).",
                "matched_segments": motion_res.get("matched_segments", [])
            }

    # 6. Compare Scene (Structural Video Cut Pattern Matching)
    scene_pct = 0.0
    scene_res = {}
    if config["features"]["scene"]["enabled"]:
        scene_res = compare_scene_structures(
            v1.get('scene', {}).get('scenes', []),
            v2.get('scene', {}).get('scenes', [])
        )
        scene_pct = scene_res.get("match_percentage", 0.0)
        dup_pct = config["features"]["scene"]["duplicate_threshold_pct"]
        if scene_pct >= dup_pct:
            return {
                "status": "duplicate",
                "match_type": "scene_structure",
                "match_percentage": scene_pct,
                "message": f"High scene edit structure similarity ({scene_pct}% >= threshold {dup_pct}%).",
                "matched_segments": scene_res.get("matched_scenes", [])
            }

    # If no method meets its threshold, find the highest match to report
    all_percentages = {
        "blake3": 0.0,
        "phash": phash_pct,
        "ai_embedding": ai_pct,
        "audio": audio_pct,
        "motion": motion_pct,
        "scene_structure": scene_pct
    }
    
    best_method = max(all_percentages, key=all_percentages.get)
    best_percentage = all_percentages[best_method]
    
    best_segments = []
    if best_method == "phash" and phash_res:
        best_segments = phash_res.get("matched_segments", [])
    elif best_method == "ai_embedding" and ai_res:
        best_segments = ai_res.get("matched_segments", [])
    elif best_method == "audio" and audio_res:
        best_segments = audio_res.get("matched_segments", [])
    elif best_method == "motion" and motion_res:
        best_segments = motion_res.get("matched_segments", [])
    elif best_method == "scene_structure" and scene_res:
        best_segments = scene_res.get("matched_scenes", [])
        
    return {
        "status": "not_duplicate",
        "message": f"No duplicate detected above the configured thresholds. Highest similarity method: {best_method} ({best_percentage}%).",
        "best_match_method": best_method,
        "best_match_percentage": best_percentage,
        "matched_segments": best_segments,
        "all_match_percentages": all_percentages
    }