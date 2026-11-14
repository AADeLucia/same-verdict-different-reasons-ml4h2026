# Same Verdict, Different Reasons

Code and results for **Same Verdict, Different Reasons: LLM-as-a-Judge and Clinician
Disagreement on Medical Chatbot Completeness** (ML4H 2026).

Alexandra DeLucia, Heyuan Huang\*, Sonal Joshi\*, Mahsa Yarmohammadi, Ahmed Hassoon, Mark Dredze
(Johns Hopkins University; \*equal contribution)

Can an LLM Judge tell when a patient-facing medical response is *incomplete*? This
toolkit (`medscore_completeness`) grades medical generative AI responses for completeness
against clinician annotations, using two completeness frameworks (omission-harm and
criteria-based) across five LLM Judges, and classifies whether a Judge's reasons match the
clinician's.

## Setup

```bash
git lfs install                # the released results are stored with Git LFS
git clone https://github.com/AADeLucia/same-verdict-different-reasons-ml4h2026.git
cd same-verdict-different-reasons-ml4h2026
conda env create -f environment.yml && conda activate medic-complete
pip install -e code/
cp .env.example .env          # then fill in PROJECT_DIR and your endpoint
python -m medscore_completeness.scripts.download_datasets
```

`PROJECT_DIR` is required — `grade_responses.py` reads it at import time.

## Released results

`results/` holds the outputs the paper's numbers come from, gzipped and stored with Git LFS,
at the paths `notebooks/analysis.py` reads. Decompress them in place to regenerate the
figures and tables without regrading or reclassifying:

```bash
find results -name "*.jsonl.gz" -exec gunzip -k {} \;
```

| Path (under `results/reframe/`) | Contents |
|---|---|
| `<judge>/MedExpert/MedExpert_combined.jsonl` | Graded MedExpert responses for each of the five Judges (step 1) |
| `tables_media/qualitative_analysis/reasoning_alignment_v5/` | Final TP reasoning-alignment labels (Yes / Partially / No; `omissions_aligned_derived`) |
| `tables_media/qualitative_analysis/disagreement_analysis_d1/` | Final FP/FN disagreement labels (`disagreement_label_derived`) |
| `tables_media/qualitative_analysis/rq2_analysis_results.jsonl` | Earlier RQ2 classifier output, the row index the final labels are merged onto. The two system prompts are stored once in `rq2_analysis_results_system_prompts.json` instead of on every row |
| `tables_media/qualitative_analysis/ml4h_blind/` | Blinded human validation labels for the RQ2 classifier (40 TP items) and their key |
| `tables_media/qualitative_analysis/annotated/`, `rq2_qualitative_validation_annotator*.jsonl` | Human validation of the earlier failure-mode taxonomy |
| `tables_media/ml4h_rebuttal/medexpert_double_annotations.jsonl` | MedExpert items with two clinician annotations (clinician–clinician agreement) |

The Judge directories are `meta-llama/Llama-3.3-70B-Instruct`, `gpt-5-mini-2025-08-07`,
`Qwen/Qwen3.6-35B-A3B`, `prometheus-eval/prometheus-7b-v2.0` and
`aaditya/Llama3-OpenBioLLM-70B` (appendix only).

## What the paper reports

| | |
|---|---|
| Dataset | MedExpert, N=530 (540 pairs, 10 held out for few-shot), 34% incomplete |
| Judges (main) | Llama 3.3-70B Instruct, GPT-5 Mini, Qwen3.6-35B-A3B, Prometheus-2 7B |
| Judge (appendix) | OpenBioLLM-70B |
| RQ2 classifier | GPT-5.4 Mini (`gpt-5.4-mini-2026-03-17`), temperature 0 |

HealthBench is used only as a retrieval pool for criteria generation (N=1,281 filtered
to English, patient-facing, single-turn), not as a reported dataset.

## Metric names

The paper's display names map to these `--metrics` keys:

| Paper | `--metrics` key |
|---|---|
| Omission-Harm Rubric (FS) | `completeness-medexpert-fss` |
| Omission-Harm Rubric (ZS) — appendix | `completeness-medexpert` |
| Criteria-Checklist | `completeness-icl-healthbench` |
| Omission-Harm Rubric, Prometheus prompt | `completeness-medexpert-prometheus` |
| Clinician annotation | `completeness-human` |

## Reproducing the results

With the released results decompressed, only step 3 is needed. Steps 1 and 2 regenerate
those results from scratch.

### 1. Grade responses

```bash
PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" \
python -m medscore_completeness.scripts.grade_responses \
  --dataset MedExpert \
  --dataset-kwargs '{"n_examples": -1}' \
  --metrics completeness-medexpert-fss completeness-icl-healthbench \
  --output-dir results/reframe/${MODEL}/MedExpert \
  --config config.yaml
```

Repeat per judge, setting `MODEL`, `MAX_TOKENS` and `DECODING_TEMPERATURE` in `.env`
(Llama & Prometheus 4096, OpenBioLLM 1024, GPT-5 Mini 8192, Qwen3.6 32768; temperature 0
except GPT-5 Mini at 1.0). Prometheus-2 7B uses `completeness-medexpert-prometheus` and
is excluded from the Criteria-Checklist.

**Produces:** `MedExpert_<metric>.jsonl` per metric plus `MedExpert_combined.jsonl`.
For the cluster, `run_grading.sh` loops all five judges — set `--account` and `SERVER` first.

### 2. RQ2 reasoning alignment and disagreement labels

This stage round-trips through the analysis notebook. Run step 3 once to export the
per-pair rows, classify them, then re-run step 3 to pick up the labels:

```bash
# a) analysis.py writes results/reframe/tables_media/qualitative_analysis/rq2_qualitative.jsonl

# b) classify each pair (async by default; --batch submits to the OpenAI Batch API, --resume polls and retrieves)
Q=results/reframe/tables_media/qualitative_analysis
PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" \
python -m medscore_completeness.scripts.reasoning_alignment \
  --input $Q/rq2_qualitative.jsonl \
  --output-dir $Q/reasoning_alignment_v5 --output-file reasoning_alignment_v5-zs_t0.jsonl \
  --model-name gpt-5.4-mini-2026-03-17 --temperature 0 --max-tokens 8192 --api-key $OPENAI_API_KEY
PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" \
python -m medscore_completeness.scripts.disagreement_analysis \
  --input $Q/rq2_qualitative.jsonl \
  --output-dir $Q/disagreement_analysis_d1 --output-file disagreement_analysis_d1_t0.jsonl \
  --model-name gpt-5.4-mini-2026-03-17 --temperature 0 --max-tokens 8192 --api-key $OPENAI_API_KEY
```

`reasoning_alignment` keeps the True Positive rows and labels Judge–clinician alignment
(Yes / Partially / No). The prompt is defined in the script (`SYSTEM_PROMPT`,
`USER_TEMPLATE`) and the final label is derived in code (`omissions_aligned_derived`).
`disagreement_analysis` keeps the FP/FN rows, matches concerns with the same rules (prompts
defined in the script, one per verdict type) and derives a disagreement label in code
(`disagreement_label_derived`). Shared code is in `rq2_common.py`. The model defaults to
`$MODEL`; the paper used `gpt-5.4-mini-2026-03-17`. Use `--subset-file` to classify only
the validation rows.

To score a classifier run against the blinded human labels, use
`tools/score_reasoning_alignment.py` (TP alignment) and `tools/score_disagreement.py`
(FP/FN labels); each script's docstring gives its usage.

### 3. Figures and tables

```bash
PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" python notebooks/analysis.py
```

A [marimo](https://marimo.io) notebook that also runs as a plain script; use
`marimo edit notebooks/analysis.py` to explore interactively. Writes to
`results/reframe/tables_media/` (the rebuttal-era analyses go to its `ml4h_rebuttal/`
subdirectory).

| Paper float | File |
|---|---|
| Table 1, discrimination stats | `discrimination_stats_medexpert.tex` |
| RQ2 omission coverage | `rq2_omission_coverage.pdf` |
| Matched-pair alignment by rubric | `ml4h_rebuttal/alignment_matched_pairs.tex` |
| Reasoning alignment by rubric (classifier and blinded-human labels) | `alignment_combined.tex` |
| Score distributions | `completeness_human_model_boxplot.pdf` |
| Threshold sensitivity | `threshold_sensitivity_medexpert.pdf` |
| Omission severity | `completeness_omission_severity.pdf` |
| Full discrimination stats (incl. OpenBioLLM) | `discrimination_stats_full.tex` |
| Inter-judge correlation | `inter_model_score_correlation.pdf` |
| Paired definition comparison (TOST) | `ml4h_rebuttal/paired_definition_tost.tex` |
| Breakdown by domain | `breakdown_domain.tex` |
| Clinician vs. Judge agreement | `ml4h_rebuttal/clinician_vs_judge_agreement.tex` |
| FP adjudication | `ml4h_rebuttal/fp_adjudication.tex` |
| Generated criteria examples | `ml4h_rebuttal/cc_generated_criteria_examples.tex` |
| RQ2 disagreement labels | `rq2_rq3_failure_modes_combined.pdf` |
| RQ2 classifier validation | `rq2_blind_validation_combined.tex` |
| Earlier taxonomy validation | `rq2_taxonomy_validation.tex`, `rq2_taxonomy_validation_heatmap.pdf` |

Response lengths are counted with the Llama 3.3 tokenizer, which is gated on Hugging Face.
Either accept its license and set `HF_TOKEN`, or point `LENGTH_TOKENIZER` at an ungated copy
of the same tokenizer:

```bash
LENGTH_TOKENIZER=unsloth/Llama-3.3-70B-Instruct PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" python notebooks/analysis.py
```

`analysis.py` needs the full graded set (released in `results/`, or step 1). The
`examples/` directory holds ten records per judge to show the output schema.

## Layout

```
code/medscore_completeness/   grader library + CLI scripts
  graders/                    Grader base class and one module per rubric
  scripts/grade_responses.py  main entry point
  scripts/reasoning_alignment.py, disagreement_analysis.py, rq2_common.py   RQ2 classifiers
notebooks/analysis.py         all paper figures and tables
tools/                        RQ2 pipeline helpers, classifier scoring, blinded annotation tool
results/                      released graded outputs and classifier labels (Git LFS, gzipped)
examples/                     sample graded outputs (10 records per judge)
rq2_rq3_details/              earlier RQ2 taxonomy definitions and validation sample
config.yaml                   per-grader parameters (!env interpolation)
run_grading.sh                SLURM driver for step 1
```

## Library usage

Any OpenAI-compatible endpoint works. Graders with fixed rubrics need only `question`
and `answer`; `completeness-icl-healthbench` additionally needs a `criteria` list.

```python
import asyncio
from medscore_completeness.graders import Grader

records = [{
    "id": "example-001",
    "question": "What are the symptoms of Type 2 diabetes?",
    "answer": "Common symptoms include increased thirst, frequent urination, and fatigue.",
    "meta": {"dataset": "custom"},
}]

grader = Grader.get_grader("completeness-medexpert-fss", {
    "model_name": "meta-llama/Llama-3.3-70B-Instruct",
    "model_server": "http://localhost:8000/v1",
    "api_key": "EMPTY",
})

results = asyncio.run(grader(records))
scoring = results[0]["completeness-medexpert-fss"]
print(scoring["score"], "/", scoring["total_points"])
```

`Grader.get_available_graders()` lists all registered names. To load parameters from
`config.yaml` instead, use `medscore_completeness.utils.configure_graders.load_config`.

## Citation

```bibtex
@inproceedings{delucia2026sameverdict,
  title     = {Same Verdict, Different Reasons: {LLM}-as-a-Judge and Clinician Disagreement on Medical Chatbot Completeness},
  author    = {DeLucia, Alexandra and Huang, Heyuan and Joshi, Sonal and Yarmohammadi, Mahsa and Hassoon, Ahmed and Dredze, Mark},
  booktitle = {Proceedings of the Machine Learning for Health (ML4H) Symposium},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). MedExpert and HealthBench are subject to
their own licenses; the released results contain MedExpert questions and responses.
