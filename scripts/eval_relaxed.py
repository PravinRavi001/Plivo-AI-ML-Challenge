import json
import argparse
import sys
from collections import defaultdict

# Ensure src is importable when running this script from the repo root
from pathlib import Path
base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base / 'src'))

from labels import label_is_pii


def load_gold(path):
    gold = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            gold[str(obj['id'])] = [(e['start'], e['end'], e['label']) for e in obj.get('entities', [])]
    return gold


def load_pred(path):
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    pred = {str(k): [(int(e['start']), int(e['end']), e['label']) for e in v] for k, v in obj.items()}
    return pred


def compute_prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def spans_overlap(a, b):
    # a and b are (s,e,lab)
    return max(a[0], b[0]) < min(a[1], b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', required=True)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)

    # Strict exact-match evaluation (same as original)
    labels = set()
    for spans in gold.values():
        for _, _, lab in spans:
            labels.add(lab)

    tp_strict = defaultdict(int)
    fp_strict = defaultdict(int)
    fn_strict = defaultdict(int)

    for uid in gold.keys():
        g_spans = set(gold.get(uid, []))
        p_spans = set(pred.get(uid, []))

        for span in p_spans:
            if span in g_spans:
                tp_strict[span[2]] += 1
            else:
                fp_strict[span[2]] += 1
        for span in g_spans:
            if span not in p_spans:
                fn_strict[span[2]] += 1

    print('Strict exact-match Macro-F1:')
    macro_sum = 0.0
    cnt = 0
    for lab in sorted(labels):
        p, r, f1 = compute_prf(tp_strict[lab], fp_strict[lab], fn_strict[lab])
        print(f'{lab:15s} P={p:.3f} R={r:.3f} F1={f1:.3f}')
        macro_sum += f1; cnt += 1
    print('Macro-F1: %.3f' % (macro_sum / max(1, cnt)))

    # Relaxed overlap-based evaluation: a predicted span counts as TP if it overlaps any gold span of same label
    tp_rel = defaultdict(int)
    fp_rel = defaultdict(int)
    fn_rel = defaultdict(int)

    # For counting, we'll keep track of matched gold spans per uid to avoid double-counting
    for uid in gold.keys():
        g_spans = gold.get(uid, [])
        p_spans = pred.get(uid, [])

        matched_g = set()

        for p in p_spans:
            matched = False
            for i, g in enumerate(g_spans):
                if g[2] == p[2] and spans_overlap(p, g):
                    tp_rel[p[2]] += 1
                    matched_g.add(i)
                    matched = True
                    break
            if not matched:
                fp_rel[p[2]] += 1

        for i, g in enumerate(g_spans):
            if i not in matched_g:
                fn_rel[g[2]] += 1

    print('\nRelaxed overlap-match Macro-F1:')
    macro_sum = 0.0
    cnt = 0
    for lab in sorted(labels):
        p, r, f1 = compute_prf(tp_rel[lab], fp_rel[lab], fn_rel[lab])
        print(f'{lab:15s} P={p:.3f} R={r:.3f} F1={f1:.3f}')
        macro_sum += f1; cnt += 1
    print('Macro-F1: %.3f' % (macro_sum / max(1, cnt)))

    # PII-only relaxed metrics
    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0

    for uid in gold.keys():
        g_spans = gold.get(uid, [])
        p_spans = pred.get(uid, [])

        matched_g = set()
        for p in p_spans:
            matched = False
            for i, g in enumerate(g_spans):
                if label_is_pii(g[2]) == label_is_pii(p[2]) and spans_overlap(p, g):
                    if label_is_pii(p[2]):
                        pii_tp += 1
                    else:
                        non_tp += 1
                    matched = True
                    matched_g.add(i)
                    break
            if not matched:
                if label_is_pii(p[2]):
                    pii_fp += 1
                else:
                    non_fp += 1
        for i, g in enumerate(g_spans):
            if i not in matched_g:
                if label_is_pii(g[2]):
                    pii_fn += 1
                else:
                    non_fn += 1

    p, r, f1 = compute_prf(pii_tp, pii_fp, pii_fn)
    print(f'\nPII-only relaxed metrics: P={p:.3f} R={r:.3f} F1={f1:.3f}')
    p2, r2, f12 = compute_prf(non_tp, non_fp, non_fn)
    print(f'Non-PII relaxed metrics: P={p2:.3f} R={r2:.3f} F1={f12:.3f}')


if __name__ == '__main__':
    main()
