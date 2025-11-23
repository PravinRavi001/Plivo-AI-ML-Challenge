import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
gold_path = base / 'data' / 'dev_gen.jsonl'
pred_path = base / 'out' / 'dev_pred.json'

print('Gold path exists:', gold_path.exists())
print('Pred path exists:', pred_path.exists())

gold = {}
with open(gold_path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        gold[str(obj['id'])] = obj.get('entities', [])

with open(pred_path, 'r', encoding='utf-8') as f:
    pred = json.load(f)

print('\nGold keys sample (first 10):')
for i, k in enumerate(sorted(list(gold.keys()))[:10]):
    print(i+1, k, type(k))

print('\nPred keys sample (first 10):')
for i, k in enumerate(sorted(list(pred.keys()))[:10]):
    print(i+1, k, type(k))

# Intersection info
gold_keys = set(gold.keys())
pred_keys = set(pred.keys())
print('\nCounts: gold=%d pred=%d intersection=%d' % (len(gold_keys), len(pred_keys), len(gold_keys & pred_keys)))

# Print first 5 ids that are in pred but not in gold, and vice versa
print('\nIn pred but not gold (sample 10):')
for i, k in enumerate(sorted(list(pred_keys - gold_keys))[:10]):
    print(i+1, k)

print('\nIn gold but not pred (sample 10):')
for i, k in enumerate(sorted(list(gold_keys - pred_keys))[:10]):
    print(i+1, k)

# Print a few sample gold vs pred spans for a matching id
common = sorted(list(gold_keys & pred_keys))
print('\nSample comparisons for first 5 matching ids:')
for k in common[:5]:
    print('\nID:', k)
    print('Gold entities:', gold[k][:5])
    print('Pred entities:', pred.get(k, [])[:5])

# Print an example where prediction had spans to inspect coordinates
print('\nExamples where pred has non-empty spans (first 5):')
count = 0
for k in sorted(pred.keys()):
    if pred[k]:
        print('\nID:', k)
        print('Text snippet:')
        # try to read the text from dev_gen
        # load corresponding text
        # fallback if not found
        # load the full dev file only once
        break

print('\nDone')
