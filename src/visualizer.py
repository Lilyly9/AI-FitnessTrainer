"""
Visualization module for exercise rep counting results.

Provides two display modes:
  - TerminalDisplay: live terminal status + final summary table
  - MatplotlibVisualizer: static charts (rep detection, bar chart, timeline)
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from rep_counter import SENSOR_COLS
    from data_utils import CLASS_NAMES
except ImportError:
    from src.rep_counter import SENSOR_COLS
    from src.data_utils import CLASS_NAMES

# 动态生成足够多的颜色（循环使用基础色盘）
_BASE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                '#1abc9c', '#e67e22', '#2980b9', '#c0392b', '#8e44ad']
CLASS_COLORS = {
    name: _BASE_COLORS[i % len(_BASE_COLORS)]
    for i, name in enumerate(CLASS_NAMES)
}


# ─── Terminal Display ────────────────────────────────────────────────

class TerminalDisplay:
    """Live terminal status display with color support."""

    def __init__(self, use_colors=True, width=72):
        self.use_colors = use_colors and sys.stdout.isatty()
        self.width = width
        self._start_time = time.time()

    @staticmethod
    def _color(code, text):
        """Wrap text with ANSI color code."""
        return f"\033[{code}m{text}\033[0m"

    def _green(self, t):
        return self._color('32', t) if self.use_colors else t

    def _yellow(self, t):
        return self._color('33', t) if self.use_colors else t

    def _red(self, t):
        return self._color('31', t) if self.use_colors else t

    def _bold(self, t):
        return self._color('1', t) if self.use_colors else t

    def update(self, exercise_name, rep_count, confidence, **kwargs):
        """Print a one-line live status update."""
        elapsed = time.time() - self._start_time
        conf_pct = confidence * 100

        if confidence > 0.7:
            conf_str = self._green(f"{conf_pct:.1f}%")
        elif confidence > 0.4:
            conf_str = self._yellow(f"{conf_pct:.1f}%")
        else:
            conf_str = self._red(f"{conf_pct:.1f}%")

        bar = self._rep_bar(rep_count, max_len=20)
        line = (f"\r[{self._bold(exercise_name):<18s}] "
                f"Reps: {self._bold(str(rep_count)):>3s} {bar} "
                f"| Conf: {conf_str} "
                f"| Time: {elapsed:.1f}s ")
        sys.stdout.write(line.ljust(self.width))
        sys.stdout.flush()

    def render_summary(self, exercise_counts, total_reps, duration_seconds):
        """Print a formatted summary table."""
        print("\n")
        header = "REP COUNTING SUMMARY"
        width = 56
        print("┌" + "─" * (width - 2) + "┐")
        print("│" + header.center(width - 2) + "│")
        print("├" + "─" * 20 + "┬" + "─" * 10 + "┬" + "─" * 11 + "┬" + "─" * 10 + "┤")

        # Header row
        hdr = (f"│ {'Exercise':<18s} │ {'Reps':>6s} │ {'Pct':>7s} │ {'Bar':<8s} │")
        print(hdr)
        print("├" + "─" * 20 + "┼" + "─" * 10 + "┼" + "─" * 11 + "┼" + "─" * 10 + "┤")

        sorted_items = sorted(exercise_counts.items(), key=lambda x: -x[1])
        for ex_name, count in sorted_items:
            pct = count / total_reps * 100 if total_reps > 0 else 0
            bar = "█" * min(int(pct / 5), 8)
            print(f"│ {ex_name:<18s} │ {count:>6d} │ {pct:>6.1f}% │ {bar:<8s} │")

        print("├" + "─" * 20 + "┼" + "─" * 10 + "┼" + "─" * 11 + "┼" + "─" * 10 + "┤")
        print(f"│ {'TOTAL':<18s} │ {total_reps:>6d} │ {'100.0%':>7s} │ {'':<8s} │")
        print("└" + "─" * 20 + "┴" + "─" * 10 + "┴" + "─" * 11 + "┴" + "─" * 10 + "┘")
        print(f"Duration: {duration_seconds:.1f} seconds")
        print()

    def show_alert(self, message, level='info'):
        """Print a timestamped alert message."""
        elapsed = time.time() - self._start_time
        prefix = {'info': '[INFO]', 'warn': '[WARN]', 'error': '[ERR]'}.get(level, '[INFO]')

        if level == 'warn':
            prefix = self._yellow(prefix)
        elif level == 'error':
            prefix = self._red(prefix)

        sys.stdout.write("\r" + " " * self.width + "\r")
        print(f"{prefix} [{elapsed:.1f}s] {message}")
        sys.stdout.flush()

    @staticmethod
    def _rep_bar(count, max_len=20):
        """ASCII bar showing rep count progress."""
        blocks = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
        full = count // 8
        remainder = count % 8
        return '█' * min(full, max_len) + ('▏' if remainder > 0 and full < max_len else '')


# ─── Matplotlib Visualizer ───────────────────────────────────────────

class MatplotlibVisualizer:
    """Generate static matplotlib charts for rep counting results."""

    def __init__(self, style='default', dpi=150, class_colors=None):
        plt.style.use(style)
        self.dpi = dpi
        self.class_colors = class_colors or CLASS_COLORS

    def plot_rep_detection(self, signal_dict, save_path=None):
        """Plot raw/filtered signal with detected peaks marked.

        Args:
            signal_dict: keys: raw, filtered, peaks, exercise_name, time_axis, sensor_data
            save_path: output file path, or None to show
        """
        raw_signal = signal_dict.get('composite', signal_dict.get('raw'))
        filtered = signal_dict['filtered']
        peaks = signal_dict['peaks']
        exercise_name = signal_dict.get('exercise_name', 'unknown')

        fig, axes = plt.subplots(2, 1, figsize=(14, 7))

        # Time axis
        t = signal_dict.get('time_axis')
        if t is None:
            t = np.arange(len(raw_signal))

        # Top: raw + filtered + peaks
        ax = axes[0]
        ax.plot(t, raw_signal, color='gray', alpha=0.4, linewidth=0.8, label='Raw')
        ax.plot(t, filtered, color='#3498db', linewidth=1.2, label='Filtered')
        if len(peaks) > 0:
            color = self.class_colors.get(exercise_name, '#e74c3c')
            ax.scatter(t[peaks], filtered[peaks], color=color, s=60, zorder=5,
                       edgecolors='white', linewidth=0.5)
            for i, p in enumerate(peaks):
                ax.annotate(str(i + 1), (t[p], filtered[p]),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=8, ha='center', color=color)

        ax.set_title(f'Rep Detection: {exercise_name} ({len(peaks)} reps)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylabel('Amplitude (normalized)')
        ax.grid(True, alpha=0.3)

        # Bottom: raw 6-channel overlay
        ax = axes[1]
        # Plot each channel normalized for reference
        if 'sensor_data' in signal_dict and signal_dict['sensor_data'] is not None:
            raw = signal_dict['sensor_data']
            for c in range(min(6, raw.shape[1])):
                ch = raw[:, c]
                ch_norm = (ch - ch.mean()) / (ch.std() + 1e-8)
                ax.plot(t, ch_norm + c * 2.5, alpha=0.5, linewidth=0.5,
                        label=SENSOR_COLS[c] if c < len(SENSOR_COLS) else f'ch{c}')
            ax.legend(loc='upper right', fontsize=7, ncol=2)

        ax.set_xlabel('Sample')
        ax.set_ylabel('Channel (offset)')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return fig

    def plot_summary_barchart(self, exercise_counts, total_reps, save_path=None):
        """Horizontal bar chart showing rep count per exercise."""
        names = list(exercise_counts.keys())
        counts = list(exercise_counts.values())
        colors = [self.class_colors.get(n, '#95a5a6') for n in names]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))

        bars = ax.barh(names, counts, color=colors, edgecolor='white', height=0.6)
        for bar, count in zip(bars, counts):
            pct = count / total_reps * 100 if total_reps > 0 else 0
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{count} ({pct:.1f}%)', va='center', fontsize=10)

        ax.set_xlabel('Repetitions')
        ax.set_title(f'Exercise Rep Count Summary (Total: {total_reps})')
        ax.set_xlim(0, max(counts) * 1.2 + 1 if counts else 10)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return fig

    def plot_timeline(self, timeline_data, save_path=None):
        """Cumulative rep count over time, colored by exercise.

        Args:
            timeline_data: list of (timestamp_s, exercise_name, cumulative_reps, rep_count)
            save_path: output file path
        """
        if not timeline_data:
            return None

        fig, ax = plt.subplots(figsize=(12, 5))

        # Build step plot with color transitions
        prev_name = None
        start_t = timeline_data[0][0]
        for i, (ts, ex_name, cum_reps, count) in enumerate(timeline_data):
            color = self.class_colors.get(ex_name, '#95a5a6')
            if ex_name != prev_name:
                label = ex_name
                prev_name = ex_name
            else:
                label = None
            ax.plot(ts - start_t, cum_reps, marker='.', color=color,
                    markersize=3, linewidth=1.2, label=label)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Cumulative Repetitions')
        ax.set_title('Cumulative Reps Over Time')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return fig
