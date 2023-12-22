import argparse
import os
import sys

from tools.runpod_audioprocessing import *

sys.path.append(os.getcwd())


def get_f0m(audio_file_path: str) -> None:
    freq = AudioProcessingUtils.get_fundamental_frequency(audio_file_path)
    print(f'Fundamental Frequency: {round(freq, 2)}Hz')


def get_semitone_distance(pitch1: float, pitch2: float) -> None:
    dist = AudioProcessingUtils.semitone_distance(pitch1, pitch2)
    print(f'Semitone Distance: {dist}')


def main():
    parser = argparse.ArgumentParser(description='Some useful audio processing operations')
    subparsers = parser.add_subparsers(dest='action', help='Available actions')

    f0m_parser = subparsers.add_parser('get-f0m', help='Get fundamental frequency from the audio file')
    f0m_parser.add_argument("audio_file_path", type=str, help="Path to audio file")

    semitone_parser = subparsers.add_parser('get-semitone-distance', help='Get semitone distance between 2 pitches')
    semitone_parser.add_argument('p1', type=float, help='Pitch 1')
    semitone_parser.add_argument('p2', type=float, help='Pitch 2')

    # Parse the command-line arguments
    args = parser.parse_args()
    if args.action == 'get-f0m':
        get_f0m(args.audio_file_path)
    elif args.action == 'get-semitone-distance':
        get_semitone_distance(args.p1, args.p2)


if __name__ == '__main__':
    main()
