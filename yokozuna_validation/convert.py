"""
Yokozuna Format Converter
Usage: python yokozuna/convert.py <input.wav> [output.ykz] [analysis_duration_seconds]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from format.encoder import YokozunaEncoder


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python yokozuna/convert.py <input.wav> [output.ykz] [duration_seconds]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].replace('.', '').isdigit() else None
    duration = None
    for arg in sys.argv[2:]:
        try:
            duration = float(arg)
            break
        except ValueError:
            continue

    encoder = YokozunaEncoder(n_modes=32, partition_depth=32)
    result = encoder.encode(input_path, output_path=output_path, analysis_duration=duration)
    print(f"\nConverted: {result}")
