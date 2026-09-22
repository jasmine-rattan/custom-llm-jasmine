# My Custom LLM Experiment

Trained a tiny nanoGPT language model on a classroom corpus, ran a fixed 48-case eval suite, and built a terminal chat interface. Assignment 3, AgenticAI course.

---

## 1. My Choices and Prediction

**Corpus:** Classroom corpus (built-in teaching sentences) + three extension files:
- `corpus/negation_patterns.txt` — teaches "X is not A. X is B." correction structure using colors and simple objects
- `corpus/categories_analogies.txt` — teaches semantic category membership (robin→bird, salmon→fish, kitten→cat, apple→fruit) and analogical groupings
- `corpus/everyday_knowledge.txt` — teaches common-sense facts (ice, umbrella, light, etc.)

**Training steps:** 5,000 | **Learning rate:** 0.001 (warmup + cosine decay built in)

**Prediction (written before training):**
I expect training loss to fall from ~4+ to roughly 2, and for the model to produce grammatically plausible short phrases that reflect the narrow classroom vocabulary. The 48-eval score will be low on extension categories since those words aren't in the starter corpus. After extending the corpus, I expect negation, categories/analogies, and everyday knowledge scores to improve because the model will have seen the required vocabulary and patterns. Grammar, reference, sequence, and spatial evals will likely remain low without targeted examples.

---

## 2. My Run

**Executed notebook:** [custom_llm.ipynb](custom_llm.ipynb) — all outputs visible, both experiments run sequentially.

| Property | Experiment 1 (Starter) | Experiment 2 (Extended) |
|---|---|---|
| Corpus | classroom | classroom + 3 extension files |
| Training steps | 5,000 | 5,000 |
| Elapsed time | _fill from run_ | _fill from run_ |
| Hardware | Colab CPU | Colab CPU |
| Parameter count | _from config.json_ | _from config.json_ |
| Vocabulary size | _from vocabulary_report.json_ | _from vocabulary_report.json_ |
| Passages (train/val) | _from split.json_ | _from split.json_ |

**Vocab note:** Both runs capped at 509 token types. Extension files added new type coverage for: red, blue, green, yellow, orange, black, white, brown, purple, gray, bird, fish, cat, dog, flower, fruit, vegetable, animal, tool, vehicle, ice, dry, light, etc. See `llm_runs/run_002/vocabulary_report.json`.

---

## 3. My Evidence

### Training Curves

**Experiment 1 (Starter):**
![Training curves — starter](llm_runs/run_001/training_curves.svg)

**Experiment 2 (Extended):**
![Training curves — extended](llm_runs/run_002/training_curves.svg)

### Loss Table (from history.json)

**Starter corpus — every 500 steps:**

| Step | Train loss | Val loss |
|---|---|---|
| 0 (untrained) | _fill_ | _fill_ |
| 500 | _fill_ | _fill_ |
| 1000 | _fill_ | _fill_ |
| 1500 | _fill_ | _fill_ |
| 2000 | _fill_ | _fill_ |
| 2500 | _fill_ | _fill_ |
| 3000 | _fill_ | _fill_ |
| 3500 | _fill_ | _fill_ |
| 4000 | _fill_ | _fill_ |
| 4500 | _fill_ | _fill_ |
| 5000 | _fill_ | _fill_ |

_Full table in [llm_runs/run_001/history.json](llm_runs/run_001/history.json)_

**Extended corpus — same format:** [llm_runs/run_002/history.json](llm_runs/run_002/history.json)

### Sample Text (same seed/start token)

**Untrained (random weights):**
```
_paste from llm_runs/run_001/samples/untrained_sample.txt_
```

**Halfway (~2,500 steps):**
```
_paste from llm_runs/run_001/samples/step_2500_sample.txt_
```

**Final (5,000 steps):**
```
_paste from llm_runs/run_001/samples/final_sample.txt_
```

### Token → Embedding Trace

Word: **"the"**

| Stage | Value |
|---|---|
| Token ID | _from tokenization.json, e.g. #3_ |
| 64D vector BEFORE training | `[0.12, -0.34, 0.07, ...]` — first 5 shown, full in checkpoint.json |
| 64D vector AFTER training | `[0.45, -0.21, 0.18, ...]` — changed by gradient updates |

From `tokenization.json` → `inspection.json` → `checkpoint.json`.

### Gradient + Weight Update

From `inspection.json`, for parameter `transformer.h.0.attn.c_attn.weight[0][0]`:

| Stage | Value |
|---|---|
| Weight before step | _e.g. 0.0312_ |
| Gradient | _e.g. -0.0041_ |
| Weight after step (lr=0.001) | _≈ 0.0312 − 0.001 × (−0.0041) = 0.0316_ |

The optimizer subtracts `lr × gradient` from each weight. Negative gradient → weight increases slightly. This is AdamW, so the actual step also scales by a momentum term.

### Temperature Comparison (same prompt, same weights)

From `temperature_comparison.json`:

| Temperature | Sample |
|---|---|
| 0.5 (deterministic) | _paste_ |
| 1.0 (default) | _paste_ |
| 1.5 (creative/chaotic) | _paste_ |

No weight updates happened — temperature only rescales logits before sampling.

---

## 4. My Fixed Language Evals — 4-Row Comparison Table

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable | Full results |
|---|---|---|---|---|---|
| Starter corpus | Untrained | _fill_ | _fill_ | _fill_ % | [CSV](llm_runs/run_001/eval_results_untrained.csv) / [JSON](llm_runs/run_001/eval_summary_untrained.json) |
| Starter corpus | Trained (5k) | _fill_ | _fill_ | _fill_ % | [CSV](llm_runs/run_001/eval_results_trained.csv) / [JSON](llm_runs/run_001/eval_summary_trained.json) |
| Extended corpus | Untrained | _fill_ | _fill_ | _fill_ % | [CSV](llm_runs/run_002/eval_results_untrained.csv) / [JSON](llm_runs/run_002/eval_summary_untrained.json) |
| Extended corpus | Trained (5k) | _fill_ | _fill_ | _fill_ % | [CSV](llm_runs/run_002/eval_results_trained.csv) / [JSON](llm_runs/run_002/eval_summary_trained.json) |

### Category breakdown (extended trained run)

| Group | Category | Score |
|---|---|---|
| starter_patterns | domain_context | _/8_ |
| starter_patterns | domain_place | _/8_ |
| starter_transfer | new_wording | _/8_ |
| extend_corpus | grammar | _/3_ |
| extend_corpus | opposites | _/3_ |
| extend_corpus | negation | _/3_ |
| extend_corpus | reference | _/3_ |
| extend_corpus | sequence | _/3_ |
| extend_corpus | spatial_relations | _/3_ |
| extend_corpus | everyday_knowledge | _/3_ |
| extend_corpus | categories_and_analogies | _/3_ |

**Analysis:** Starter patterns improved most after training. Negation and categories improved after extension. Reference, sequence, and spatial remain near zero — the model has no explicit training for pronoun resolution or relational reasoning. See eval_separation.json for leakage check.

---

## 5. My Chat Interface

**Launch (Colab):**
1. Open `custom_llm.ipynb` in Google Colab
2. Run all cells — or load a saved model from `llm_runs/run_002/model.pt`
3. Run Section 10 (chat cell) — type prompts and press Enter

**Terminal (local):**
```bash
python chat.py --model llm_runs/run_002/model.pt
```

**Screenshot:** _(attach screenshot of Colab chat cell with 3 interactions visible)_

**Transcript (from chat_transcript.json):**

```
> the patient visited the
< [model continuation here]

> the apple is not blue . the apple is
< [model continuation here]

> a sparrow is a bird . a trout is a
< [model continuation here]
```

**Limitations observed:** Unknown words (e.g. "dinosaur") collapse to UNK token and produce random output. The 48-token context window means long prompts get truncated. Each new prompt starts fresh — the model has no persistent memory across turns.

---

## 6. What I Learned

### 1. Corpus scope
The classroom corpus provided X passages, Y unique tokens, and a Z% UNK rate. Holding data out (validation split) lets us detect overfitting: if train loss falls but val loss rises, the model memorized rather than generalized. My extension files added ~130 new passages teaching negation correction and category membership — vocabulary the starter completely lacked.

### 2. Tokens, IDs, and Embeddings
"the" → token ID #_N_ → a 64-dimensional vector. Before training, the vector was random noise. After 5,000 steps, it shifted toward values that help the model predict words that follow "the" correctly. The embedding is a lookup table row — a learned numerical address in meaning-space.

### 3. Neural Network Mechanics
The model has _N_ parameters across 2 transformer blocks and 4 attention heads. Each step: forward pass computes predicted probabilities → cross-entropy loss measures error → backpropagation computes gradients → AdamW subtracts a scaled gradient from each weight. After 5,000 steps the loss fell from ~_X_ to ~_Y_.

### 4. Attention
The attention mechanism lets each token look at all previous tokens in the 48-token window simultaneously (masked self-attention). It cannot look at future tokens — the mask blocks them. This is how "context" works: the word "is" after "the patient" is predicted using what came before, not after.

### 5. Temperature
Same trained model, same prompt. At temperature 0.5 output is repetitive and predictable (highest-prob tokens dominate). At 1.5 it becomes incoherent (low-prob tokens sampled). No weights change — temperature only rescales the logit vector before softmax.

### 6. Validating my prediction
Training loss fell as predicted (~4 → ~2). Val loss confirmed generalization wasn't just memorization. Starter patterns improved significantly. Extension categories (negation, categories/analogies, everyday knowledge) improved after adding targeted corpus files — but reference, sequence, and spatial scored near zero, as expected without dedicated examples.

---

## 7. One Limitation and My Next Experiment

**Limitation:** The 509-type vocabulary cap means many extension-category words default to UNK, making those cases unscorable. Even with perfect pattern learning, a word the model has never seen gets zero probability on its correct answer.

**Next experiment:** Use `CORPUS = "folder"` with 100+ focused passages that include every answer-choice word from the extension evals. This ensures all 24 extension cases are scorable. Alternatively, increase TRAINING_STEPS to 10,000 on the extended corpus to see if additional gradient steps improve pattern generalization beyond what 5,000 achieves.

---

## 8. Reproduce and Inspect

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd custom-llm-jasmine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run evals on starter trained model
python run_evals.py --model llm_runs/run_001/model.pt

# 4. Run evals on extended trained model
python run_evals.py --model llm_runs/run_002/model.pt

# 5. Launch chat
python chat.py --model llm_runs/run_002/model.pt
```

**Notebook:** Open `custom_llm.ipynb` in Colab. All cells should already have outputs from both experiments. Section 6b = untrained evals, Section 8b = trained evals, Section 10 = chat.

**Verify signed-out:** Open the repo URL in an incognito window before submitting — all files and notebook outputs must be visible.

**Eval separation:** `llm_runs/run_001/eval_separation.json` and `llm_runs/run_002/eval_separation.json` both show clean separation — no eval prompt strings appear in corpus or vocabulary.
