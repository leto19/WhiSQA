"""
Command-line interface for WhiSQA.
"""

import argparse
import sys
from pathlib import Path

from .core import get_score, get_score_multi


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WhiSQA: Whisper-based Speech Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisqa --audio_file audio.wav --model_type single
  whisqa --audio_files *.wav --model_type multi --batch_size 8
  whisqa --audio_file audio.wav --checkpoint_dir ./my_checkpoints
        """
    )
    
    # Audio input options
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--audio_file", 
        type=str, 
        help="Path to a single audio file"
    )
    audio_group.add_argument(
        "--audio_files", 
        type=str, 
        nargs='+', 
        help="List of paths to audio files for batch processing"
    )
    
    # Model options
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["single", "multi"],
        default="single",
        help="Model type: 'single' for MOS score, 'multi' for multidimensional scores"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for processing multiple files (default: 16)"
    )
    
    parser.add_argument(
        "--checkpoint_dir", 
        type=str,
        help="Directory containing model checkpoints (optional)"
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["human", "json", "csv"],
        default="human",
        help="Output format (default: human)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.audio_file:
            # Single file processing
            score = get_score(args.audio_file, args.model_type, args.checkpoint_dir)
            _print_single_result(args.audio_file, score, args.model_type, args.output_format)
        
        elif args.audio_files:
            # Batch processing
            scores = get_score_multi(args.audio_files, args.model_type, args.batch_size, args.checkpoint_dir)
            _print_batch_results(args.audio_files, scores, args.model_type, args.output_format)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _print_single_result(audio_file: str, score, model_type: str, output_format: str):
    """Print results for a single audio file."""
    if output_format == "json":
        import json
        result = {"file": audio_file}
        if model_type == "single":
            result["mos"] = float(score.item() * 5)
        else:
            result.update({
                "mos": float(score[0].item() * 5),
                "noisiness": float(score[1].item() * 5),
                "coloration": float(score[2].item() * 5),
                "discontinuity": float(score[3].item() * 5),
                "loudness": float(score[4].item() * 5)
            })
        print(json.dumps(result, indent=2))
    
    elif output_format == "csv":
        if model_type == "single":
            print("file,mos")
            print(f"{audio_file},{score.item() * 5:.3f}")
        else:
            print("file,mos,noisiness,coloration,discontinuity,loudness")
            print(f"{audio_file},{score[0].item() * 5:.3f},{score[1].item() * 5:.3f},"
                  f"{score[2].item() * 5:.3f},{score[3].item() * 5:.3f},{score[4].item() * 5:.3f}")
    
    else:  # human format
        print(f"File: {audio_file}")
        if model_type == "single":
            print(f"MOS: {score.item() * 5:.3f}")
        else:
            print(f"MOS: {score[0].item() * 5:.3f}")
            print(f"Noisiness: {score[1].item() * 5:.3f}")
            print(f"Coloration: {score[2].item() * 5:.3f}")
            print(f"Discontinuity: {score[3].item() * 5:.3f}")
            print(f"Loudness: {score[4].item() * 5:.3f}")


def _print_batch_results(audio_files: list, scores, model_type: str, output_format: str):
    """Print results for multiple audio files."""
    if output_format == "json":
        import json
        results = []
        for audio_file, score in zip(audio_files, scores):
            result = {"file": audio_file}
            if model_type == "single":
                result["mos"] = float(score.item())
            else:
                result.update({
                    "mos": float(score[0].item()),
                    "noisiness": float(score[1].item()),
                    "coloration": float(score[2].item()),
                    "discontinuity": float(score[3].item()),
                    "loudness": float(score[4].item())
                })
            results.append(result)
        print(json.dumps(results, indent=2))
    
    elif output_format == "csv":
        if model_type == "single":
            print("file,mos")
            for audio_file, score in zip(audio_files, scores):
                print(f"{audio_file},{score.item():.3f}")
        else:
            print("file,mos,noisiness,coloration,discontinuity,loudness")
            for audio_file, score in zip(audio_files, scores):
                print(f"{audio_file},{score[0].item():.3f},{score[1].item():.3f},"
                      f"{score[2].item():.3f},{score[3].item():.3f},{score[4].item():.3f}")
    
    else:  # human format
        for audio_file, score in zip(audio_files, scores):
            print(f"\nFile: {audio_file}")
            if model_type == "single":
                print(f"MOS: {score.item():.3f}")
            else:
                print(f"MOS: {score[0].item():.3f}")
                print(f"Noisiness: {score[1].item():.3f}")
                print(f"Coloration: {score[2].item():.3f}")
                print(f"Discontinuity: {score[3].item():.3f}")
                print(f"Loudness: {score[4].item():.3f}")


if __name__ == "__main__":
    main()