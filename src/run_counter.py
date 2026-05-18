"""
Rep Counter CLI — exercise rep counting and visualization entry point.

Usage:
  # File mode + auto exercise detection
  python src/run_counter.py --input data/raw/gym_gesture/imu_dataset.csv \\
      --mode file --auto --save-charts

  # File mode + manual exercise
  python src/run_counter.py --input data/raw/gym_gesture/imu_dataset.csv \\
      --mode file --exercise chest_press

  # Streaming mode (simulated from file)
  python src/run_counter.py --input data/raw/gym_gesture/imu_dataset.csv \\
      --mode stream --auto

  # Fast-forward streaming (no delay)
  python src/run_counter.py --input data/raw/gym_gesture/imu_dataset.csv \\
      --mode stream --auto --stream-speed 0 --no-terminal
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

try:
    from rep_counter import RepCounter, read_raw_csv, chunk_generator, SENSOR_COLS
    from exercise_classifier import ExerciseClassifier, normalize_sensor_data, CLASS_NAMES
    from visualizer import TerminalDisplay, MatplotlibVisualizer
except ImportError:
    from src.rep_counter import RepCounter, read_raw_csv, chunk_generator, SENSOR_COLS
    from src.exercise_classifier import ExerciseClassifier, normalize_sensor_data, CLASS_NAMES
    from src.visualizer import TerminalDisplay, MatplotlibVisualizer


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser(
        description='IMU-based Exercise Repetition Counter & Visualizer')
    p.add_argument('--input', required=True,
                   help='Path to raw IMU CSV file')
    p.add_argument('--mode', choices=['file', 'stream'], default='file',
                   help='Processing mode: file=batch, stream=simulated real-time')
    p.add_argument('--exercise', default=None,
                   help='Manual exercise name (overrides auto-detection)')
    p.add_argument('--auto', action='store_true',
                   help='Enable model-based automatic exercise recognition')
    p.add_argument('--use-labels', action='store_true',
                   help='Use ground-truth labels from CSV to split into exercise segments')
    p.add_argument('--output', default='results/counts/',
                   help='Output directory for charts and JSON results')
    p.add_argument('--no-terminal', action='store_true',
                   help='Disable terminal live output')
    p.add_argument('--save-charts', action='store_true',
                   help='Save matplotlib charts to output directory')
    p.add_argument('--save-json', action='store_true',
                   help='Save rep count results as JSON')
    p.add_argument('--chunk-size', type=int, default=50,
                   help='Samples per chunk in streaming mode (default: 50)')
    p.add_argument('--stream-speed', type=float, default=1.0,
                   help='Speed multiplier for streaming (0=fast-forward, 1=real-time)')
    p.add_argument('--filter-cutoff', type=float, default=2.0,
                   help='Butterworth low-pass cutoff in Hz (default: 2.0)')
    p.add_argument('--min-peak-distance', type=int, default=0,
                   help='Min samples between peaks, 0=auto from autocorrelation (default: 0)')
    p.add_argument('--prominence', type=float, default=0.2,
                   help='Peak prominence in std units (default: 0.2)')
    p.add_argument('--fs', type=int, default=100,
                   help='Sampling frequency in Hz (default: 100)')
    p.add_argument('--athlete', type=int, default=None,
                   help='Filter to specific athlete_id for per-subject analysis')
    p.add_argument('--model-dir', default='models/',
                   help='Directory containing trained model weights')
    return p.parse_args()


def run_file_mode(args):
    """Batch processing: read file, classify (if auto), count reps, visualize."""
    print(f"Reading: {args.input}")
    sensor_data, labels, athlete_ids = read_raw_csv(args.input)
    print(f"  Total samples: {len(sensor_data)}")

    if len(sensor_data) == 0:
        print("Error: No sensor data found.")
        return 1

    # Filter by athlete if requested
    if args.athlete is not None and athlete_ids is not None:
        mask = athlete_ids == args.athlete
        sensor_data = sensor_data[mask]
        if labels is not None:
            labels = labels[mask]
        print(f"  Filtered to athlete {args.athlete}: {len(sensor_data)} samples")

    # Initialize components
    terminal = None if args.no_terminal else TerminalDisplay()
    mpl_viz = MatplotlibVisualizer() if args.save_charts else None
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Determine exercise segments
    segments = []  # list of (start_idx, end_idx, exercise_name)

    if args.exercise:
        # Manual exercise — uses whole file
        if args.exercise not in CLASS_NAMES:
            print(f"Warning: '{args.exercise}' not in known classes {CLASS_NAMES}")
        segments.append((0, len(sensor_data), args.exercise))
        print(f"Manual mode: '{args.exercise}' — whole file")

    elif args.use_labels and labels is not None:
        # Use ground-truth labels to split by exercise
        print("Using ground-truth labels to split segments...")
        ex_names = np.unique(labels)
        for ex in ex_names:
            idxs = np.where(labels == ex)[0]
            if len(idxs) == 0:
                continue
            # Find contiguous blocks
            starts = [idxs[0]]
            ends = []
            for i in range(1, len(idxs)):
                if idxs[i] - idxs[i-1] > 500:  # gap > 5 sec = new segment
                    ends.append(idxs[i-1] + 1)
                    starts.append(idxs[i])
            ends.append(idxs[-1] + 1)
            for s, e in zip(starts, ends):
                if e - s >= 200:  # minimum segment length
                    segments.append((s, e, str(ex)))
        print(f"  Found {len(segments)} segments across {len(ex_names)} exercises")

    elif args.auto:
        # Auto-detect exercises via model
        print("Loading classifier...")
        classifier = ExerciseClassifier(
            model_dir=args.model_dir, class_names=CLASS_NAMES, use_ensemble=True)
        classifier.load_models()

        # Normalize data for classifier
        normalized_data, min_v, max_v = normalize_sensor_data(sensor_data)
        # Save normalization stats
        stats_path = os.path.join(args.output, 'normalization_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({'min': min_v.tolist(), 'max': max_v.tolist()}, f, indent=2, cls=NpEncoder)
        print(f"Normalization stats saved to {stats_path}")

        # Classify segments — split by exercise transitions
        result = classifier.classify_segment(normalized_data)
        majority_class = result['class_name']
        confidence = result['confidence']
        print(f"Detected exercise: {majority_class} (confidence: {confidence:.1%})")
        segments.append((0, len(sensor_data), majority_class))

    else:
        print("Error: Specify --exercise, --auto, or --use-labels.")
        return 1

    # Step 2: Count reps per segment
    rep_counter = RepCounter(
        filter_cutoff=args.filter_cutoff, fs=args.fs,
        min_peak_distance=args.min_peak_distance, prominence=args.prominence)

    all_results = []
    exercise_totals = defaultdict(int)
    timeline = []

    for seg_idx, (start, end, ex_name) in enumerate(segments):
        seg_data = sensor_data[start:end]
        count, peaks, filtered, raw = rep_counter.count_reps(seg_data)

        exercise_totals[ex_name] += count
        seg_result = {
            'segment': seg_idx,
            'start_sample': start,
            'end_sample': end,
            'exercise': ex_name,
            'rep_count': count,
            'duration_seconds': (end - start) / args.fs,
        }
        all_results.append(seg_result)

        # Terminal output
        if terminal:
            duration = (end - start) / args.fs
            terminal.update(ex_name, count, 1.0)
            terminal.show_alert(
                f"{ex_name}: {count} reps detected ({duration:.1f}s)")

        # Save rep detection chart
        if mpl_viz:
            chart_path = os.path.join(args.output, f'rep_detection_{ex_name}_{seg_idx}.png')
            mpl_viz.plot_rep_detection({
                'raw': raw,
                'filtered': filtered,
                'peaks': peaks,
                'exercise_name': ex_name,
                'time_axis': np.arange(start, end)[:len(raw)] / args.fs,
                'sensor_data': seg_data,
            }, save_path=chart_path)
            print(f"  Chart saved: {chart_path}")

    # Step 3: Summary
    total_reps = sum(exercise_totals.values())
    total_duration = len(sensor_data) / args.fs

    if terminal:
        terminal.render_summary(dict(exercise_totals), total_reps, total_duration)

    if mpl_viz:
        bar_path = os.path.join(args.output, 'summary_barchart.png')
        mpl_viz.plot_summary_barchart(
            dict(exercise_totals), total_reps, save_path=bar_path)
        print(f"Summary chart saved: {bar_path}")

    # Step 4: JSON output
    if args.save_json:
        json_path = os.path.join(args.output, 'rep_count_results.json')
        output = {
            'input_file': args.input,
            'mode': 'file',
            'total_samples': len(sensor_data),
            'total_reps': total_reps,
            'duration_seconds': total_duration,
            'sampling_frequency': args.fs,
            'exercise_counts': dict(exercise_totals),
            'segments': all_results,
            'params': {
                'filter_cutoff': args.filter_cutoff,
                'min_peak_distance': args.min_peak_distance,
                'prominence': args.prominence,
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)
        print(f"JSON results saved: {json_path}")

    return 0


def run_stream_mode(args):
    """Simulated real-time streaming: read file in chunks, classify + count, visualize."""
    print(f"Reading: {args.input}")
    sensor_data, labels, athlete_ids = read_raw_csv(args.input)
    print(f"  Total samples: {len(sensor_data)}")

    if len(sensor_data) == 0:
        print("Error: No sensor data found.")
        return 1

    # Filter by athlete if requested
    if args.athlete is not None and athlete_ids is not None:
        mask = athlete_ids == args.athlete
        sensor_data = sensor_data[mask]
        if labels is not None:
            labels = labels[mask]
        print(f"  Filtered to athlete {args.athlete}: {len(sensor_data)} samples")

    # Initialize components
    terminal = None if args.no_terminal else TerminalDisplay()
    mpl_viz = MatplotlibVisualizer() if args.save_charts else None
    os.makedirs(args.output, exist_ok=True)

    rep_counter = RepCounter(
        filter_cutoff=args.filter_cutoff, fs=args.fs,
        min_peak_distance=args.min_peak_distance, prominence=args.prominence)

    current_exercise = args.exercise or 'unknown'
    confidence = 1.0 if args.exercise else 0.0

    # Load classifier if auto mode
    classifier = None
    if args.auto:
        print("Loading classifier...")
        classifier = ExerciseClassifier(
            model_dir=args.model_dir, class_names=CLASS_NAMES, use_ensemble=True)
        classifier.load_models()

    # Create data generators
    raw_gen = chunk_generator(sensor_data, chunk_size=args.chunk_size,
                              speed=args.stream_speed, fs=args.fs)

    if args.auto and classifier:
        normalized_data, min_v, max_v = normalize_sensor_data(sensor_data)
        norm_gen = chunk_generator(normalized_data, chunk_size=args.chunk_size,
                                   speed=0, fs=args.fs)  # no double sleep
        class_gen = classifier.classify_stream(norm_gen)
    else:
        class_gen = None

    # Main streaming loop
    timeline = []
    exercise_counts = defaultdict(int)
    prev_exercise = current_exercise

    raw_iter = iter(raw_gen)
    class_iter = iter(class_gen) if class_gen else None

    session_start = time.time()

    while True:
        try:
            chunk, is_end = next(raw_iter)
        except StopIteration:
            break

        # Get classification result
        if class_iter:
            try:
                class_result = next(class_iter)
                new_exercise = class_result.get('class_name', 'unknown')
                if new_exercise != current_exercise:
                    prev_exercise = current_exercise
                    current_exercise = new_exercise
                    confidence = class_result.get('confidence', 0.0)
                    if terminal and not class_result.get('final'):
                        terminal.show_alert(
                            f"Exercise changed: {prev_exercise} -> {current_exercise}",
                            level='info')
            except StopIteration:
                pass

        # Count reps
        rep_result = rep_counter.count_reps_streaming([(chunk, is_end)])
        try:
            rep_data = next(rep_result)
        except StopIteration:
            continue
        except RuntimeError:
            # Generator already consumed
            break

        new_peaks = rep_data.get('new_peaks', 0)
        if new_peaks > 0:
            exercise_counts[current_exercise] += new_peaks

        # Terminal display
        if terminal:
            total_reps = sum(exercise_counts.values())
            terminal.update(current_exercise, total_reps, confidence)

        # Timeline
        elapsed = time.time() - session_start
        timeline.append((elapsed, current_exercise,
                        sum(exercise_counts.values()), new_peaks))

        if is_end:
            break

    # Final summary
    total_reps = sum(exercise_counts.values())
    total_duration = len(sensor_data) / args.fs

    if terminal:
        terminal.render_summary(dict(exercise_counts), total_reps, total_duration)

    # Charts
    if mpl_viz and timeline:
        tl_path = os.path.join(args.output, 'timeline.png')
        mpl_viz.plot_timeline(timeline, save_path=tl_path)
        print(f"Timeline chart saved: {tl_path}")

        bar_path = os.path.join(args.output, 'summary_barchart.png')
        mpl_viz.plot_summary_barchart(
            dict(exercise_counts), total_reps, save_path=bar_path)
        print(f"Summary chart saved: {bar_path}")

    # JSON
    if args.save_json:
        json_path = os.path.join(args.output, 'rep_count_results.json')
        output = {
            'input_file': args.input,
            'mode': 'stream',
            'total_samples': len(sensor_data),
            'total_reps': total_reps,
            'duration_seconds': total_duration,
            'sampling_frequency': args.fs,
            'exercise_counts': dict(exercise_counts),
            'params': {
                'filter_cutoff': args.filter_cutoff,
                'min_peak_distance': args.min_peak_distance,
                'prominence': args.prominence,
                'chunk_size': args.chunk_size,
                'stream_speed': args.stream_speed,
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)
        print(f"JSON results saved: {json_path}")

    return 0


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    print("=" * 55)
    print("IMU Repetition Counter & Visualizer")
    print("=" * 55)
    print(f"Input: {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Detection: {'auto' if args.auto else f'manual ({args.exercise})'}")
    print()

    if args.mode == 'file':
        ret = run_file_mode(args)
    else:
        ret = run_stream_mode(args)

    print("\nDone.")
    return ret


if __name__ == '__main__':
    sys.exit(main())
