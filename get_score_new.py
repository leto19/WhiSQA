#!/usr/bin/env python3
"""
Legacy CLI script for backward compatibility.
This script imports from the new whisqa package.
"""

import sys
import argparse
from pathlib import Path

# Add the package to the path for development
sys.path.insert(0, str(Path(__file__).parent))

try:
    from whisqa.core import get_score, get_score_multi
except ImportError:
    # Fallback for development/local usage
    from whisqa.core import get_score, get_score_multi


def main():
    """Main entry point - maintains backward compatibility."""
    parser = argparse.ArgumentParser(description="Get a score for a given audio file")
    parser.add_argument("--audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--audio_files", type=str, nargs='+', help="List of paths to audio files for batch processing")
    parser.add_argument("--model_type", type=str, help="Single headed MOS or multidimension [MOS,Noisiness, Coloration,Discontinuity and Loudness]", default="single")
    args = parser.parse_args()

    try:
        if args.audio_file:
            score = get_score(args.audio_file, args.model_type)
            print(f"Score for {args.audio_file}: {score}")
            
            # Print in the old format for backward compatibility
            if args.model_type == "single":
                print("MOS", score.item() * 5)
            else:
                mos = score[0].item() * 5
                noisiness = score[1].item() * 5
                coloration = score[2].item() * 5
                discontinuity = score[3].item() * 5
                loudness = score[4].item() * 5
                print("MOS", mos)
                print("Noisiness", noisiness)
                print("Coloration", coloration)
                print("Discontinuity", discontinuity)
                print("Loudness", loudness)

        if args.audio_files:
            scores = get_score_multi(args.audio_files, args.model_type)
            for audio_file, score in zip(args.audio_files, scores):
                print(f"Score for {audio_file}: {score}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()