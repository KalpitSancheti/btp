#!/usr/bin/env python3
"""
draw_graph.py -- improved logging

Usage:
  python draw_graph.py --csv "sac_training_logs/training_log.csv" --n 1 --begin 0

This script logs:
 - which CSV it used
 - how many rows were parsed
 - lengths of each metric
 - saved PNG paths
 - any parsing errors encountered

Logs are printed to stdout and also written to <out_dir>/plot.log
"""
import os
import csv
import glob
import argparse
import logging
import sys
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

plt.switch_backend('Agg')  # headless

def setup_logging(out_dir, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'plot.log')
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, mode='a', encoding='utf-8')]
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=handlers)
    logging.info('Logging started. log file: %s', log_path)

def block_smooth(arr, n):
    """Non-overlapping block average (safe: auto-adjusts n)."""
    arr = np.asarray(arr, dtype=float)
    L = len(arr)
    if L == 0:
        return np.array([], dtype=float)
    # ensure n is at least 1 and not bigger than length (we allow n == L -> one point)
    n = max(1, min(int(n), L))
    if n == 1:
        return arr.copy()
    end = -(L % n)
    trimmed = arr if end == 0 else arr[:end]
    if trimmed.size == 0:
        # fallback to single average
        return np.array([np.mean(arr)], dtype=float)
    reshaped = trimmed.reshape(-1, n)
    return np.mean(reshaped, axis=1)

def moving_average(arr, window):
    if window <= 1:
        return np.array(arr, dtype=float)
    weights = np.ones(window, dtype=float) / window
    return np.convolve(arr, weights, mode='valid')

def draw_series(name, x, series, n=100):
    x = np.array(x, dtype=float)
    s = np.array(series, dtype=float)
    # auto-adjust n to target reasonable number of output points
    # but keep behavior consistent with passed n (we clamp)
    n_use = max(1, min(n, len(s))) if len(s) > 0 else 1

    x_smooth = block_smooth(x, n_use)
    s_smooth = block_smooth(s, n_use)

    plt.plot(x_smooth, s_smooth, label=name, linewidth=2.4)

    if len(s_smooth) > 0:
        cumavg = np.cumsum(s_smooth) / (np.arange(len(s_smooth)) + 1)
        plt.plot(x_smooth, cumavg, color='gray', linestyle='--', linewidth=1.8,
                 label=f'{name} cumulative avg')

        rolling_win = min(50, max(1, len(s_smooth) // 10))
        if rolling_win > 1 and len(s_smooth) >= rolling_win:
            roll = moving_average(s_smooth, rolling_win)
            x_roll = x_smooth[(rolling_win - 1):]
            plt.plot(x_roll, roll, linestyle=':', linewidth=1.8,
                     label=f'{name} moving avg (w={rolling_win})')

def find_csv_paths():
    candidates = [
        './training_log.csv',
        './training_graph.csv',
        './training_log.csv',
        './sac_training_logs/training_log.csv',
        './save_stat/training_log.csv',
        './save_stat/training_graph.csv',
    ]
    candidates += glob.glob('./save_stat/*_stat.csv')
    candidates += glob.glob('./training*.csv')
    # unique and existing
    seen = []
    for p in candidates:
        if p and p not in seen and os.path.exists(p):
            seen.append(p)
    return seen

def parse_csv_generic(path):
    """
    Flexibly parse CSV. Returns:
    episodes, bestY, getgoal, step, step_per_record, step_for_goal, rows_parsed
    """
    logging.info('Parsing CSV: %s', path)
    bestY = []
    step = []
    getgoal = []
    step_per_record = []
    step_for_goal = []

    rows_parsed = 0
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            first = None
            for row in reader:
                if row and any(cell.strip() for cell in row):
                    first = row
                    break
            if first is None:
                logging.warning('CSV is empty or contains only empty rows: %s', path)
                return [], [], [], [], [], [], 0

            # detect header vs numeric first row
            numeric_count = 0
            for c in first:
                try:
                    float(c)
                    numeric_count += 1
                except Exception:
                    pass

            # rewind
            f.seek(0)
            reader = csv.reader(f)

            header_map = {}
            if numeric_count < len(first):
                # header present (non-numeric found)
                header_row = next(reader)
                cols = [c.strip().lower() for c in header_row]
                for i, col in enumerate(cols):
                    if col in ('t','time','step','steps','duration','episode_time'):
                        header_map['time'] = i
                    if col in ('besty','best_y','best','best_record','bestscore','best_score'):
                        header_map['besty'] = i
                    if col in ('goal','is_goal','getgoal','reached_goal'):
                        header_map['is_goal'] = i
                    if col in ('episode','ep'):
                        header_map['episode'] = i
                logging.info('Detected header with columns: %s', cols)
            else:
                # no header; fallback indices match your original parser
                header_map['time'] = 1
                header_map['besty'] = 3
                logging.info('No header detected. Using fallback indices: time=1, besty=3')

            prev_step_for_goal = 0
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                try:
                    t = None
                    besty = None
                    # time
                    if 'time' in header_map and header_map['time'] < len(row):
                        t = float(row[header_map['time']])
                    else:
                        # fallback: first numeric
                        for c in row:
                            try:
                                t = float(c)
                                break
                            except:
                                continue
                    # besty
                    if 'besty' in header_map and header_map['besty'] < len(row):
                        besty = float(row[header_map['besty']])
                    else:
                        try:
                            besty = float(row[3]) if len(row) > 3 else None
                        except:
                            besty = None

                    if t is None or besty is None:
                        # not enough numeric info; skip row
                        continue

                    bestY.append(besty)
                    step.append(t)
                    is_goal = 1 if besty >= 57 else 0
                    getgoal.append(is_goal)
                    sfg = t if is_goal else prev_step_for_goal
                    step_for_goal.append(sfg)
                    prev_step_for_goal = sfg
                    step_per_record.append(t / besty if besty > 1 else t)
                    rows_parsed += 1
                except Exception as e:
                    logging.debug('Skipping row due to parse error: %s -- %s', row, e)
                    continue
    except Exception as e:
        logging.exception('Failed to open/parse CSV %s: %s', path, e)
        return [], [], [], [], [], [], 0

    logging.info('CSV parsed: %d rows', rows_parsed)
    episodes = list(range(len(bestY)))
    return episodes, bestY, getgoal, step, step_per_record, step_for_goal, rows_parsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100, help='block smoothing window (use 1 for no smoothing)')
    parser.add_argument('--begin', type=int, default=0, help='use only last <begin> episodes (0=all)')
    parser.add_argument('--out-dir', type=str, default='save_graph', help='directory to save generated graphs')
    parser.add_argument('--csv', type=str, default=None, help='explicit CSV file to use')
    parser.add_argument('--verbose', action='store_true', help='verbose logging')
    args = parser.parse_args()

    setup_logging(args.out_dir, verbose=args.verbose)

    csv_path = args.csv
    if csv_path:
        if not os.path.exists(csv_path):
            logging.error('Provided CSV not found: %s', csv_path)
            logging.info('Attempting auto-detect...')
            csv_path = None

    if not csv_path:
        found = find_csv_paths()
        if not found:
            logging.error('No CSV found in common locations. Provide path with --csv')
            return
        csv_path = found[0]
        logging.info('Auto-detected CSV: %s', csv_path)

    episodes, bestY, getgoal, step, step_per_record, step_for_goal, rows_parsed = parse_csv_generic(csv_path)
    if rows_parsed == 0 or not episodes:
        logging.error('No data parsed from CSV, exiting.')
        return

    ylabels = [
        'Best Record', 'Get Goal Prob.', 'Step', 'Step per Record', 'Step for Goal'
    ]
    metrics = [bestY, getgoal, step, step_per_record, step_for_goal]

    for idx, ylabel in enumerate(ylabels):
        plt.figure(figsize=(18, 4.5))
        plt.title(ylabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=14)
        plt.xlabel('episode', fontsize=14)

        metric_to_plot = metrics[idx]
        logging.info('Plotting metric "%s" length=%d (n=%d)', ylabel, len(metric_to_plot), args.n)
        draw_series('training', episodes, metric_to_plot, n=args.n)

        plt.grid(alpha=0.25)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                   fancybox=True, ncol=1, fontsize='large')

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_label = ylabel.replace(" ", "_")
        out_path = os.path.join(args.out_dir, f'result_{safe_label}_{ts}.png')
        try:
            plt.savefig(out_path, dpi=150, format='png')
            plt.close()
            logging.info('Saved plot: %s', out_path)
        except Exception as e:
            logging.exception('Failed to save plot %s: %s', out_path, e)

if __name__ == '__main__':
    main()
