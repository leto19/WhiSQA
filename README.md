# Whisper based Speech Quality Assessment (WhiSQA)

Usage:
`pip install .`

CLI:
`whisqa /path/to/mono_wav_file.wav`

Batch CLI:
`whisqa /path/to/first.wav /path/to/second.wav`

Legacy script entry point:
`python3 get_score.py /path/to/mono_wav_file.wav`

Python API:
`from whisqa import get_score, get_scores`

Input audio must be mono. Non-16 kHz inputs are resampled automatically to 16 kHz.

Requires `git-lfs`

![Results](results.png)
