# RQ2 / RQ3 Qualitative Analysis — Methods and File Index

> **Historical.** This file documents an earlier version of the RQ2 classifier
> (`analyze_rq2_qualitative.py`, FP01–FN05 codes). The ML4H 2026 paper reports the final
> classifiers instead: `reasoning_alignment.py` (TP alignment) and `disagreement_analysis.py`
> (FP/FN disagreement labels), both run with GPT-5.4 Mini. See the root README.

## Summary

For RQ2 and RQ3, we used GPT-5 mini (`gpt-5-mini-2025-08-07`) as a meta-judge to classify the reasoning alignment between LLM judges and clinicians across 27,135 evaluation rows. Each row was first assigned to a confusion-matrix quadrant (TP/TN/FP/FN) based on whether the LLM judge's verdict matched the clinician's. Same-verdict rows (TP/TN) were analyzed for omission alignment — whether the model and clinician identified the same missing information — using a directional, recall-based scale (Yes/Partially/No) that treats the clinician as ground truth. Different-verdict rows (FP/FN) were classified into one of nine failure mode categories capturing how the model's reasoning diverged from the clinician's. The two-prompt architecture, failure mode codebook, and few-shot examples were developed iteratively by the author using Claude Opus 4.6 as a collaborative assistant: the author provided manually annotated examples and design constraints, and Claude proposed and refined the taxonomy, prompt structure, and alignment definitions across multiple revision cycles. The final prompts were grounded in 35 manually annotated examples; a 20-row pilot (5 per quadrant, randomly sampled) confirmed prompt calibration before full-scale execution via the OpenAI Batch API.

---

## Process

### Step 1: Manual Annotation (Codebook Development)

35 examples were manually annotated by the author — spanning all four confusion-matrix quadrants across multiple rubrics, LLM judges, and datasets. These annotations established:

- The **alignment scale** for same-verdict cases: Yes / Partially / No (recall-based; clinician is ground truth)
- The **failure mode taxonomy** for different-verdict cases: FP01–FP04 and FN01–FN05

The taxonomy was derived inductively from the annotations. FN05 ("over-credits generic 'consult your doctor' as sufficient") was added after it recurred across multiple FN cases. The full development conversation is in `rq2_rq3_claude_conversation.txt`.

Key design decision: alignment is measured as **recall against the clinician's concerns**, not symmetric overlap. A model that identifies all of the clinician's concerns plus additional ones scores "Yes"; a model that misses any of the clinician's concerns scores "Partially" or "No" depending on how many it captured.

**Failure mode taxonomy:**

| Code | Description |
|------|-------------|
| FP01 | Over-flags minor gaps (core question clinically answered) |
| FP02 | Demands clinical depth beyond patient-facing scope |
| FP03 | Misapplies safety-netting criteria to a non-urgent context |
| FP04 | Misinterprets question intent |
| FN01 | Fails to detect the omission entirely |
| FN02 | Detects omission but underweights severity |
| FN03 | Detects omission but misframes clinical significance |
| FN04 | Clinician flags minor/debatable gap (arguably outside question scope) |
| FN05 | Over-credits generic "consult your doctor" as sufficient coverage |

### Step 2: Prompt Construction

Two prompts were written, one per branch:

- **Prompt 1** (TP/TN — same verdict): Extracts omissions cited by each rater and classifies alignment as Yes/Partially/No. Contains 8 few-shot examples covering all three alignment levels across both TP and TN cases.
- **Prompt 2** (FP/FN — different verdict): Classifies the model's failure mode into one of the 9 categories above, plus an `omissions_aligned` field. Contains 11 few-shot examples covering all 9 categories.

Both prompts use the chatbot answer as context (added after initial development when it was found the meta-judge needed the response to verify whether cited omissions were actually absent). The system prompt contains all instructions, definitions, and few-shot examples; the user message per row contains only the 5 variable fields (verdict type, question, answer, human judgment text, model judgment text).

Output was enforced as JSON (`response_format: json_object`), with `seed=42` and `max_completion_tokens=4096`.

### Step 3: Pilot Validation

20 rows were randomly sampled (5 per quadrant) and run through both prompts. Results were reviewed against the manual annotations. 17/20 outputs were unchanged after a directional-framing refinement to the prompts; the 3 changes were all improvements (confirmed against manual labels). Two of the 20 pilot rows share a question ID with the manually annotated set, but in both cases the (judge model, rubric) combination was different — the specific model output evaluated was not a few-shot example.

### Step 4: Full-Scale Execution

The finalized prompts were run on all 27,135 rows via the OpenAI Batch API (5,000-row chunks, 24h completion window, 60s poll interval). The main LLM judge set (LLaMA 3.3-70B, OpenBioLLM, GPT-4o mini) and Prometheus-2 7B were run in separate batches due to Prometheus-2 7B being added to the analysis later.

### Step 5: Results Analysis

Results were loaded in `notebooks/analysis.py` (§4 and §5), merged across batches, and aggregated into pivot tables and figures. Prometheus-2 7B results were merged from a separate file at load time. Legacy metric name normalization (`FSS` → `FS`) was applied on load.

---

## File Index

### Codebook and Prompt Development

| File | Description |
|------|-------------|
| `rq2_rq3_details/rq2_rq3_claude_conversation.txt` | Full transcript of the iterative codebook and prompt development session |
| `rq2_rq3_details/rq2_rq3_manual_coding_MarchARR.xlsx` | 35 manually annotated examples (TP/TN/FP/FN sheets) used to derive the codebook and few-shot examples; also used for spot-checking |
| `rq2_rq3_details/validation_20_examples.jsonl` | 20 randomly sampled pilot examples (5 per quadrant) with GPT-4o mini outputs used to validate prompts before full run |

### Prompts

| File | Description |
|------|-------------|
| `code/medscore_completeness/scripts/prompt1_same_verdict.md` | System prompt for TP/TN rows: omission alignment classification (Yes/Partially/No) with 8 few-shot examples |
| `code/medscore_completeness/scripts/prompt2_different_verdict.md` | System prompt for FP/FN rows: failure mode classification (FP01–FP04, FN01–FN05) with 11 few-shot examples |

### Execution Script

| File | Description |
|------|-------------|
| `code/medscore_completeness/scripts/analyze_rq2_qualitative.py` | Main script: prepares rows, routes to correct prompt, runs via OpenAI Batch API or async streaming, handles idempotent resume via `batch_state.json` |

### Input Data

| File | Description |
|------|-------------|
| `results/results_jan_arr_rebuttal/rq2_qualitative.jsonl` | Full export of all 27,135 rows (LLaMA, OpenBioLLM, GPT-4o mini) with confusion-matrix category labels, prepared by `analysis.py` §4.1 |
| `results/results_jan_arr_rebuttal/rq2_qualitative_prometheus7b.jsonl` | Prometheus-2 7B rows only, exported separately and run through a separate batch |

### Output Data

| File | Description |
|------|-------------|
| `results/results_jan_arr_rebuttal/rq2_analysis/rq2_analysis_results.jsonl` | Meta-judge outputs for the main three LLM judges (23,292 rows) |
| `results/results_jan_arr_rebuttal/rq2_analysis/rq2_analysis_results_prometheus7b.jsonl` | Meta-judge outputs for Prometheus-2 7B (10,767 rows) |
| `results/results_jan_arr_rebuttal/rq2_analysis/rq2_TP.jsonl` | TP rows only (intermediate, per-category split) |
| `results/results_jan_arr_rebuttal/rq2_analysis/rq2_TN.jsonl` | TN rows only |
| `results/results_jan_arr_rebuttal/rq2_analysis/rq2_FP.jsonl` | FP rows only |
| `results/results_jan_arr_rebuttal/rq2_analysis/rq2_FN.jsonl` | FN rows only |

### Analysis and Figures

| File | Description |
|------|-------------|
| `notebooks/analysis.py` §4–§5 | Loads results, merges Prometheus-2 7B, computes pivot tables, generates all RQ2/RQ3 figures and LaTeX tables |
| `paper/tables_media/rq2_omission_coverage.pdf` | Heatmap: omission alignment (Yes/Partially/No) by confusion-matrix category |
| `paper/tables_media/rq3_fp_fn_failure_modes.pdf` | Stacked bar chart: FP/FN failure mode distribution aggregated across all models and datasets |
| `paper/tables_media/rq3_fp_fn_medexpert_arfs_dc.pdf` | MedExpert-only comparison: AR (FS) vs. Dynamic-Checklist, 2×2 grid |
| `paper/tables_media/rq2_align_pivot.tex` | LaTeX table: alignment (%) by confusion-matrix category |
| `paper/tables_media/rq2_fp_pivot.tex` | LaTeX table: FP failure mode (%) by rubric |
| `paper/tables_media/rq2_fn_pivot.tex` | LaTeX table: FN failure mode (%) by rubric |
| `paper/tables_media/rq2_counts.tex` | LaTeX table: row counts per confusion-matrix category |
| `paper/tables_media/rq2_disagreements.xlsx` | Formatted export (TP/TN/FP/FN sheets) used for human spot-checking and paper examples |
