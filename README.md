# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.
Task: in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).

## My Solution
Model Selection:
I initially evaluated distilbert-base-uncased (Baseline) and albert-base-v2.

DistilBERT: ~45ms inference (Too slow).

ALBERT: ~120ms inference. Despite being "Lite" in parameter count, its 12-layer depth makes CPU inference significantly slower than DistilBERT.

Selected Model: microsoft/xtremedistil-l6-h384-uncased

Why: This model retains the 6-layer depth of DistilBERT but reduces the hidden dimension from 768 to 384. This "narrowing" reduces matrix multiplication costs by ~4x, allowing us to hit 14ms latency without quantization.

### Data Strategy: Robustness via Formatting Noise
Generating high-quality synthetic data was the primary challenge.

Initial Approach: I attempted to simulate complex STT errors (e.g., "four two" for 42). This resulted in massive span misalignment issues where the model would predict the digit but miss the word, leading to 0 precision.

Final Approach: I switched to a "Robust Digit" strategy. I generate PII as digits but inject random formatting noise (dashes, spaces, dots) within the sequences (e.g., 4-2 4 2).

Benefit: This ensures exact span alignment while forcing the model to learn token grouping rather than memorizing simple templates. This makes the model robust to the most common type of STT segmentation errors.

### Latency Optimization
To achieve the sub-20ms target on a CPU with Batch Size 1, I forced single-threaded execution using torch.set_num_threads(1). PyTorch's default multi-threading creates significant overhead ("thrashing") for such small batch sizes; disabling it reduced latency by ~40%.

### Key Hyperparameters
Epochs: 5 (Sufficient for the model to converge on the synthetic patterns).

Learning Rate: 5e-4 (Higher than standard to force the pre-trained model to aggressively adapt to the specific numeric distribution of the PII data).

Batch Size: 8 (Standard for small dataset fine-tuning).

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name microsoft/xtremedistil-l6-h384-uncased
  --train data/train_gen.jsonl
  --dev data/dev_gen.jsonl
  --out_dir out
  --epochs 5
  --lr 5e-4
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev_gen.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev_gen.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```


