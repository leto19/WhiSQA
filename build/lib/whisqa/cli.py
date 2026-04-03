import argparse
import sys

SUPPORTED_MODEL_TYPES = ("single", "multi")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Get a score for a given audio file")
    parser.add_argument("audio_file", nargs="+", type=str, help="Path to one or more audio files")
    parser.add_argument(
        "--model-type",
        "--model_type",
        dest="model_type",
        choices=sorted(SUPPORTED_MODEL_TYPES),
        default="single",
        help="Single MOS or multidimensional MOS output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from .api import batch_scores_to_dicts, get_scores

    scores = get_scores(args.audio_file, args.model_type)
    print(scores.shape)
    for audio_file, score_dict in zip(args.audio_file, batch_scores_to_dicts(scores, args.model_type)):
        if len(args.audio_file) > 1:
            print(audio_file)
        for label, value in score_dict.items():
            print(label, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
