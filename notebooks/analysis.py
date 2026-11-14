"""analysis.py

Marimo notebook for paper plot and table generation
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import os
    import random
    import typing
    import pickle
    import itertools
    import json
    import re
    import shutil
    import warnings
    import gzip

    from dotenv import load_dotenv
    import scipy as sp
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from transformers import AutoTokenizer
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support, average_precision_score, cohen_kappa_score
    import simpledorff
    import krippendorff
    import tiktoken
    # jsonlines replaced by pandas to_json(lines=True)

    # Load environment variables
    load_dotenv()

    # Set display defaults
    pd.set_option("display.float_format", '{:.2f}'.format)
    pd.set_option("display.precision", 2)
    return (
        AutoTokenizer,
        average_precision_score,
        cohen_kappa_score,
        confusion_matrix,
        gzip,
        itertools,
        json,
        krippendorff,
        matplotlib,
        mlines,
        mo,
        mpatches,
        np,
        os,
        pd,
        pickle,
        plt,
        precision_recall_fscore_support,
        random,
        re,
        roc_auc_score,
        shutil,
        simpledorff,
        sns,
        sp,
        tiktoken,
        typing,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MEDIC-Completeness Analysis

    This notebook evaluates how well automated completeness metrics align with human clinician judgments on medical Q&A datasets.

    **Research Questions:**

    - **RQ1 — Discriminative Ability:** Can LLM Judge reliably distinguish complete from incomplete patient-facing medical responses? Measured via Krippendorff's α, F1, AUC-ROC, confusion matrices, and triage utility.
    - **RQ2 — Reasoning Alignment:** Beyond verdict agreement, how well do LLM Judge and clinician reasoning align — and what failure patterns arise when verdicts diverge?
    - **RQ3 - Cross-Completeness Performance:** What patterns emerge when applying all completeness frameworks to all datasets?

    **Datasets:**
    - **MedExpert** — Real LLM responses annotated by clinicians with an omission severity level (None → Life-threatening), providing a 5-point ordinal human completeness score.
    - **HealthBench** — Benchmark Q&A pairs with expert-written rubric criteria; human completeness score derived from unweighted ICL grading.

    **Grading methods compared:**
    - `Criteria-Checklist` (`completeness-icl-healthbench`) — per-criterion binary ICL scoring; in-dataset for HealthBench
    - `Omission-Harm Rubric`* (`completeness-omission-harm`) — omission-harm rubric; in-dataset for MedExpert
    - `Omission-Harm Rubric (FS)` (`completeness-omission-harm-few-shot`) — omission-harm rubric with few-shot prompting; in-dataset for MedExpert
    - `General-Likert`* (`completeness-clinical-significance`) — general medical safety Likert (appendix-only)
    - `General-Likert (FS)`* (`completeness-clinical-significance-fss`) — general medical safety Likert with few-shot prompting (appendix-only)
    - `Clinician` (`completeness-human`) — human annotation (ground truth)

    *Appendix-only
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Settings

    Configure which models, datasets, and metrics to load. `COMPLETE_MODE` controls the binarisation threshold used throughout RQ1: `"strict"` treats only score = 1.0 as complete; `"relaxed"` uses a 0.75 threshold.
    """)
    return


@app.cell
def _(mo):
    complete_mode_widget = mo.ui.radio(
        options={
            "Relaxed  —  score ≥ 0.75  (allows minor omissions)": "relaxed",
            "Strict   —  score = 1.0   (no omissions)": "strict",
        },
        value="Relaxed  —  score ≥ 0.75  (allows minor omissions)",
        label="**Completeness threshold mode**",
    )
    mo.output.replace(complete_mode_widget)
    return (complete_mode_widget,)


@app.cell(hide_code=True)
def _(mo):
    include_health_professional_widget = mo.ui.radio(
        options={
            "Exclude  —  consumer questions only": False,
            "Include  —  all questions": True,
        },
        value="Exclude  —  consumer questions only",
        label="**Health professional questions**",
    )
    mo.output.replace(include_health_professional_widget)
    return (include_health_professional_widget,)


@app.cell
def _(complete_mode_widget, include_health_professional_widget):
    INCLUDE_HEALTH_PROFESSIONAL = include_health_professional_widget.value
    COMPLETE_MODE = complete_mode_widget.value
    return COMPLETE_MODE, INCLUDE_HEALTH_PROFESSIONAL


@app.cell
def _(os):
    SETTINGS = [
        {
            "folder": "reframe/meta-llama/Llama-3.3-70B-Instruct",
            "model": "LLaMA 3.3-70B",
            "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        },
        {
            "folder": "reframe/aaditya/Llama3-OpenBioLLM-70B",
            "model": "OpenBioLLM",
            "model_id": "aaditya/Llama3-OpenBioLLM-70B",
        },
        {
            "folder": "reframe/gpt-5-mini-2025-08-07",
            "model": "GPT-5 Mini",
            "model_id": "gpt-5-mini-2025-08-07",
        },
        {
            "folder": "reframe/Qwen/Qwen3.6-35B-A3B",
            "model": "Qwen3.6",
            "model_id": "Qwen/Qwen3.6-35B-A3B",
        },
        {
            "folder": "reframe/prometheus-eval/prometheus-7b-v2.0",
            "model": "Prometheus-2 7B",
            "model_id": "prometheus-eval/prometheus-7b-v2.0",
        },
        {
            "folder": "reframe/prometheus-eval/prometheus-bgb-8x7b-v2.0",
            "model": "Prometheus-2 BGb 8x7B",
            "model_id": "prometheus-eval/prometheus-bgb-8x7b-v2.0",
        },
    ]

    DATASET_PATH_MAP = {
        # Maps dataset display name → (folder_name, file_prefix) when they differ from the default
        # (default: folder = dataset, file_prefix = dataset)
        "HealthBench-MT": ("HealthBench_multiturn", "HealthBench"),
    }
    DATASETS = ["MedExpert", "HealthBench", "HealthBench-MT"]
    # Datasets the paper actually reports. DATASETS stays the full load set so the
    # cache stays reusable and the few cells that deliberately span datasets (the
    # malformed-output table, the parked HealthBench analyses) can still reach them.
    # Everything downstream of `results_df` inherits MAIN_DATASETS instead.
    MAIN_DATASETS = ["MedExpert"]

    # Completeness grader names → display names
    completeness_methods = {
        "completeness-criteria-checklist-healthbench": "Criteria-Checklist",
        "completeness-omission-harm": "Omission-Harm Rubric (Zero-shot)",
        "completeness-omission-harm-few-shot": "Omission-Harm Rubric",
        "completeness-medexpert-prometheus": "Omission-Harm Rubric (Prometheus)",
        "completeness-clinical-significance": "General-Likert",
        "completeness-clinical-significance-few-shot": "General-Likert (FS)",
        "completeness-medical-safety-prometheus": "General-Likert (Prometheus)",
        "completeness-human": "Clinician",
    }

    # Other evaluated metrics (metric key → display name)
    trade_off_metrics = {
        "thoroughness": "Thoroughness",
        "response-depth": "Response Depth",
        "personalization-context": "Patient Context Incorporation",
        "readability-general": "Readability (General)",
        "readability-medical": "Readability (Medical)",
        "response_length": "Response Length",
    }

    METRICS = list(completeness_methods.keys()) + [m for m in trade_off_metrics if m != "response_length"]

    # The graders were renamed after these results were produced, so every results
    # file (and the pickle cache) still carries the old keys. Map them onto the
    # current names on load; without this, nothing downstream matches
    # `completeness_methods` and the notebook has no completeness rows to work with.
    LEGACY_METRIC_ALIASES = {
        "completeness-icl-healthbench": "completeness-criteria-checklist-healthbench",
        "completeness-medexpert": "completeness-omission-harm",
        "completeness-medexpert-fss": "completeness-omission-harm-few-shot",
        "completeness-medical-safety": "completeness-clinical-significance",
        "completeness-medical-safety-fss": "completeness-clinical-significance-few-shot",
    }
    # Keys to look for when reading source files: current names plus the legacy ones.
    METRICS_ON_DISK = METRICS + list(LEGACY_METRIC_ALIASES)

    # Output directory for saved figures
    OUTPUT_DIR = f"{os.environ['PROJECT_DIR']}/results/reframe/tables_media"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # RQ2 automated analysis results
    RQ2_ANALYSIS_RESULTS_PATH = os.path.join(
        OUTPUT_DIR, "qualitative_analysis/rq2_analysis_results.jsonl"
    )
    # Final TP reasoning-alignment classifier (v5 zero-shot, label derived in code,
    # GPT-5.4 Mini at temperature 0). Its labels replace the old `omissions_aligned`
    # for MedExpert TP rows; rq2_analysis_results.jsonl is kept unchanged for comparison.
    RQ2_TP_ALIGNMENT_PATH = os.path.join(
        OUTPUT_DIR, "qualitative_analysis/reasoning_alignment_v5/reasoning_alignment_v5-zs_t0.jsonl"
    )
    # Final FP/FN disagreement classifier (d1, label derived in code; TODO §4c). Its
    # `disagreement_label_derived` becomes the `failure_mode` column for MedExpert FP/FN rows;
    # the old FP01-FN05 codes stay in `model_response_parsed['category']` for the old validation.
    RQ2_DISAGREEMENT_PATH = os.path.join(
        OUTPUT_DIR, "qualitative_analysis/disagreement_analysis_d1/disagreement_analysis_d1_t0.jsonl"
    )

    # Cache file — stores the fully-loaded & pre-processed results_df so expensive
    # file I/O and tokenisation can be skipped on subsequent notebook runs.
    CACHE_PATH = os.path.join(os.environ['PROJECT_DIR'], 'results/reframe/analysis_cache.pkl.gz')
    return (
        CACHE_PATH,
        DATASETS,
        DATASET_PATH_MAP,
        LEGACY_METRIC_ALIASES,
        MAIN_DATASETS,
        METRICS,
        METRICS_ON_DISK,
        OUTPUT_DIR,
        RQ2_ANALYSIS_RESULTS_PATH,
        RQ2_DISAGREEMENT_PATH,
        RQ2_TP_ALIGNMENT_PATH,
        SETTINGS,
        completeness_methods,
        trade_off_metrics,
    )


@app.cell(hide_code=True)
def _(DATASETS, MAIN_DATASETS, SETTINGS, completeness_methods, mo):
    _model_names = [s['model'] for s in SETTINGS]
    # Display names for all completeness metrics (including Clinician ground truth)
    _metric_display_names = list(completeness_methods.values())

    _default_models = [m for m in _model_names if m != "Prometheus-2 BGb 8x7B"]
    _default_metrics = [m for m in _metric_display_names if not m.startswith("General-Likert") and "Prometheus" not in m]

    active_models_widget = mo.ui.multiselect(
        options=_model_names,
        value=_default_models,
        label="**Active models** (plots & tables only — does not affect cache)",
    )
    active_datasets_widget = mo.ui.multiselect(
        options=DATASETS,
        value=MAIN_DATASETS,
        label="**Active datasets** (plots & tables only — does not affect cache)",
    )
    active_metrics_widget = mo.ui.multiselect(
        options=_metric_display_names,
        value=_default_metrics,
        label="**Active metrics** (plots & tables only — does not affect cache)",
    )
    mo.vstack([active_models_widget, active_datasets_widget, active_metrics_widget])
    return active_datasets_widget, active_metrics_widget, active_models_widget


@app.cell
def _(OUTPUT_DIR, matplotlib, os, plt, sns, typing):
    def save_figure(plot_object: typing.Union[sns.axisgrid.FacetGrid, matplotlib.figure.Figure, matplotlib.pyplot], base_filename: str) -> None:
        """Save a matplotlib/seaborn plot to OUTPUT_DIR as a PDF."""
        plt.rcParams['svg.fonttype'] = 'none'
        save_path = base_filename if base_filename.startswith(OUTPUT_DIR) else f'{OUTPUT_DIR}/{base_filename}'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for mode in ["pdf"]:
            _save_path = save_path + f".{mode}"
            plot_object.savefig(_save_path, format=mode, transparent=True, bbox_inches='tight')
            print(f'Saved figure to {_save_path}')

    return (save_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data Loading & Preparation

    Reads `{dataset}_combined.jsonl` for each (model, dataset) pair from the configured results directories.
    Each combined file is expanded into one row per (item, metric), with the normalised score (`points / total_points`) in the `score` column.
    Response length (tokens) is computed once per unique answer using the reference model's tokenizer.
    """)
    return


@app.cell(hide_code=True)
def _(CACHE_PATH, mo, os):
    _cache_exists = os.path.exists(CACHE_PATH)
    _status_text = (
        f"✅ Cache found: `{CACHE_PATH}`"
        if _cache_exists
        else f"⚠️ No cache found at `{CACHE_PATH}` — will load from source files."
    )

    use_cache_widget = mo.ui.checkbox(
        label="Use cached results (if available)",
        value=_cache_exists,
    )
    regen_cache_button = mo.ui.button(
        label="🔄 Regenerate cache from source files",
        kind="warn",
    )

    mo.vstack([
        mo.md(_status_text),
        use_cache_widget,
        regen_cache_button,
        mo.md(
            "_Regenerate re-loads all source JSONL files, recomputes response lengths, "
            "applies all filters, and saves a fresh pickle to the results folder. "
            "Once regenerated, check **Use cached results** and re-run to load instantly next time._"
        ),
    ])
    return regen_cache_button, use_cache_widget


@app.cell
def _(AutoTokenizer, os, pd, tiktoken):
    _tokenizer_cache: dict = {}

    def get_response_length(response: str, model_id: str) -> int:
        """Calculate response length in tokens. Caches tokenizer per model.

        Use HuggingFace tokenizer for all models except GPT-*
        """
        if 'gpt' in model_id.lower():
            key = 'gpt'
            if key not in _tokenizer_cache:
                _tokenizer_cache[key] = tiktoken.encoding_for_model('gpt-4o')
            return len(_tokenizer_cache[key].encode(response))
        else:
            if model_id not in _tokenizer_cache:
                # LENGTH_TOKENIZER points at an ungated copy of the same tokenizer
                # (e.g. unsloth/Llama-3.3-70B-Instruct) for users without gated-repo access.
                _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(os.environ.get('LENGTH_TOKENIZER', model_id))
            return len(_tokenizer_cache[model_id].encode(response, add_special_tokens=False))


    def expand_metric_results(row: pd.Series, metrics: list[str]) -> list[dict]:
        """Expand a combined-JSONL row into one dict per metric.

        Note: eval_model is NOT set here; it is applied externally from SETTINGS.
        Handles missing total_points (e.g., human grader) with fallback score = raw score.
        """
        formatted_results = []
        datapoint_metadata = row['meta'].copy()
        datapoint_metadata['dataset'] = row.dataset
        for _metric in metrics:
            metric_results = row.get(_metric)
            if pd.isna(metric_results):
                continue
            score = metric_results.get("score")
            points = metric_results.get("points")
            total_points = metric_results.get("total_points")
            try:
                # Score is either normalized float or raw points; if the former is missing, compute from the latter.
                if isinstance(score, int):
                    points = score
                    score = points / total_points
                new_row = {'id': row.name, 'metric': _metric, 'total_points': total_points, 'points': points, 'score': score, 'metric_details': metric_results.copy()}
            except (KeyError, TypeError) as err:
                new_row = {'id': row.name, 'metric': _metric}
                print(f'err={err!r} metric={_metric!r} row.name={row.name!r} metric_results={metric_results!r}')
            new_row.update(datapoint_metadata)
            formatted_results.append(new_row)
        return formatted_results


    def load_results(path: str, eval_model: str, dataset: str, metrics: list[str]) -> pd.DataFrame:
        """Load a combined JSONL results file and expand into per-metric rows."""
        if not os.path.exists(path):
            print(f'{path} not found. Skipping.')
            return pd.DataFrame()
        df = pd.read_json(path, lines=True)
        df['dataset'] = dataset
        all_rows = df.apply(lambda row: expand_metric_results(row, metrics), axis='columns').tolist()
        flattened_rows = [r for sublist in all_rows for r in sublist]
        formatted_df = pd.DataFrame(flattened_rows)
        formatted_df['eval_model'] = eval_model
        print(f'Loaded {len(df)} samples → {len(formatted_df)} metric rows from {path}')
        return formatted_df

    return get_response_length, load_results


@app.cell
def _(
    CACHE_PATH,
    COMPLETE_MODE,
    DATASETS,
    DATASET_PATH_MAP,
    INCLUDE_HEALTH_PROFESSIONAL,
    LEGACY_METRIC_ALIASES,
    METRICS_ON_DISK,
    SETTINGS,
    completeness_methods,
    get_response_length,
    gzip,
    load_results,
    mo,
    os,
    pd,
    pickle,
    regen_cache_button,
    trade_off_metrics,
    use_cache_widget,
):
    # --- Cache control ---
    # `regen_cache_button.value` increments on each click, causing this cell to re-run.
    # When the button has been clicked at least once in this session we bypass the cache
    # entirely and rebuild from source files.  The checkbox lets the user opt into the
    # cache at notebook startup (or after a regeneration that produced a fresh pickle).
    _regen_requested = (regen_cache_button.value or 0) > 0
    _use_cache = use_cache_widget.value and not _regen_requested

    if _use_cache and os.path.exists(CACHE_PATH):
        # ── Fast path: load from pickle ───────────────────────────────────────
        with gzip.open(CACHE_PATH, 'rb') as _f:
            results_df_all = pickle.load(_f)
        print(f'Loaded {len(results_df_all):,} rows from cache: {CACHE_PATH}')
        _n_legacy = results_df_all['metric'].isin(LEGACY_METRIC_ALIASES).sum()
        if _n_legacy:
            results_df_all['metric'] = results_df_all['metric'].map(
                lambda m: LEGACY_METRIC_ALIASES.get(m, m)
            )
            print(f'Renamed {_n_legacy:,} rows from legacy grader keys.')
        if 'human_judgment' not in results_df_all.columns:
            print('⚠️  Cache predates judgment columns — regenerate cache to populate human_judgment/model_judgment.')
        # Always re-derive metric_display from current completeness_methods so renames
        # are reflected without requiring a full cache rebuild.
        _disp_map = {**completeness_methods, **{k: v for k, v in trade_off_metrics.items()}}
        results_df_all['metric_display'] = results_df_all['metric'].map(lambda x: _disp_map.get(x, x))

    else:
        # ── Slow path: read source JSONL files ───────────────────────────────
        raw_dfs = []
        for setting in SETTINGS:
            results_dir = f"{os.environ['PROJECT_DIR']}/results/{setting['folder']}"
            for dataset in DATASETS:
                _folder, _file_prefix = DATASET_PATH_MAP.get(dataset, (dataset, dataset))
                results_path = f'{results_dir}/{_folder}/{_file_prefix}_combined.jsonl'
                _df = load_results(results_path, setting['model'], dataset, METRICS_ON_DISK)
                raw_dfs.append(_df)

        results_df_all = pd.concat(raw_dfs, ignore_index=True)
        results_df_all['id'] = results_df_all['id'].astype(str)
        results_df_all['metric'] = results_df_all['metric'].map(
            lambda m: LEGACY_METRIC_ALIASES.get(m, m)
        )

        # IDs used as few-shot prompt examples when developing the MedExpert FSS grader.
        # Excluded from MedExpert evaluation to prevent data leakage in FSS metrics.
        MEDEXPERT_FSS_EXCLUDED_IDS = {
            'KwqFiu6vBV3F9oFfGv9eU_WD-wfjSBQlFlxLX6Mto2Y=',
            'PpF34of5yp4vaMx7wxbPOnvH1KjZGPiRfaDRnHg2DtU=',
            'BP3ABuGWLMPw5QPQaqlEMPjakcieNgfDjum9NjP_uzY=',
            'ill11bMrTlH8bkQAWyu0MGM5zSjBYESWmHLnxCTeI3U=',
            'wlV266F4GHBTABXuSTMVDDLh3dhPUbaTW7aTA2NhAhw=',
            'BCZ0JQA3_qyILUF_X8P4O4m6I2LglDZNV_botDGUtVU=',
            '3BOHWh_9vBcGZK2HS79C3PreroixlvQ5vDfOit9iyYQ=',
            'H1BsDXnqG3IXWOeGkitJH9MT2ET-KcnBDxVlcMaIF-s=',
            't8sXoC8ahI11NQ71NLSTBivXY1R5dSLnnYET-cP7BpI=',
            'DJ_7bx_e0LzfCXNFUiUw6U00-34GxD-Or8jaeBvW1io=',
        }
        n_before_excl = len(results_df_all)
        results_df_all = results_df_all[
            ~((results_df_all['dataset'] == 'MedExpert') & (results_df_all['id'].isin(MEDEXPERT_FSS_EXCLUDED_IDS)))
        ].copy()
        print(f'Excluded {n_before_excl - len(results_df_all):,} rows (MedExpert FSS prompt examples).')

        results_df_all.set_index('id', inplace=True)
        results_df_all['has_patient_context'] = results_df_all['has_patient_context'].astype(str)
        if not INCLUDE_HEALTH_PROFESSIONAL:
            results_df_all = results_df_all[results_df_all.from_health_professional == False].copy()
        n_before = len(results_df_all)
        results_df_all = results_df_all[results_df_all['score'].notnull()].copy()
        n_after = len(results_df_all)
        if n_before > n_after:
            print(f'Dropped {n_before - n_after} rows with NaN scores.')
        _has_human = set(
            results_df_all[
                (results_df_all['metric'] == 'completeness-human') &
                results_df_all['score'].notnull()
            ]
            .reset_index()[['id', 'dataset']]
            .itertuples(index=False, name=None)
        )
        n_before = len(results_df_all)
        results_df_all = results_df_all[
            [(_id, _ds) in _has_human for _id, _ds in zip(results_df_all.index, results_df_all['dataset'])]
        ].copy()
        print(f'Kept {len(results_df_all):,} rows ({n_before - len(results_df_all):,} dropped — no human score).')
        _ref_model_id = SETTINGS[0]['model_id']
        _unique_answers = results_df_all['answer'].unique()
        _length_map = {ans: get_response_length(ans, _ref_model_id) for ans in _unique_answers}
        results_df_all['response_length'] = results_df_all['answer'].map(_length_map)
        _metric_display_map = {**completeness_methods, **{k: v for k, v in trade_off_metrics.items()}}
        results_df_all['metric_display'] = results_df_all['metric'].map(lambda x: _metric_display_map.get(x, x))

        # ── Pre-compute binary completeness judgments ─────────────────────────
        # Build (dataset, id) → human score lookup from completeness-human rows.
        _human_scores = {
            (_ds, _id): _s
            for _id, _ds, _s in (
                results_df_all[results_df_all['metric'] == 'completeness-human']
                .reset_index()[['id', 'dataset', 'score']]
                .itertuples(index=False, name=None)
            )
        }
        _label_map = {True: 'Complete', False: 'Incomplete'}

        def _human_judgment(row):
            score = _human_scores.get((row['dataset'], row.name))
            if score is None:
                return None
            return (score == 1.0) if COMPLETE_MODE == 'strict' else (score >= 0.75)

        def _model_judgment(row):
            if COMPLETE_MODE == 'strict':
                return row['score'] == 1.0
            metric = row['metric']
            if metric == 'completeness-criteria-checklist-healthbench':
                return row['score'] >= 0.75
            pts = row.get('points')
            if pts is None or (isinstance(pts, float) and pd.isna(pts)):
                return None
            if metric in ('completeness-omission-harm', 'completeness-omission-harm-few-shot', 'completeness-medexpert-prometheus'):
                return pts >= 4
            if metric in ('completeness-clinical-significance', 'completeness-clinical-significance-few-shot', 'completeness-medical-safety-prometheus'):
                return pts >= 2
            return None

        results_df_all['human_judgment'] = results_df_all.apply(_human_judgment, axis=1).map(_label_map)
        results_df_all['model_judgment'] = results_df_all.apply(_model_judgment, axis=1).map(_label_map)
        print(f'Judgment columns added: {results_df_all["human_judgment"].notna().sum():,} human, {results_df_all["model_judgment"].notna().sum():,} model judgments.')

        # ── Save cache ────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with gzip.open(CACHE_PATH, 'wb', compresslevel=3) as _f:
            pickle.dump(results_df_all, _f, protocol=pickle.HIGHEST_PROTOCOL)
        mo.output.append(mo.md(f"💾 Cache saved to `{CACHE_PATH}` ({len(results_df_all):,} rows, `COMPLETE_MODE={COMPLETE_MODE}`)."))

    print(f'\nTotal rows: {len(results_df_all)}')
    print('\nRows by eval_model x dataset:')
    print(results_df_all.groupby(['eval_model', 'dataset']).size().to_string())
    return (results_df_all,)


@app.cell
def _(
    MAIN_DATASETS,
    active_datasets_widget,
    active_metrics_widget,
    active_models_widget,
    results_df_all,
):
    # Apply model / dataset / metric filters selected in Settings.
    # `results_df_all` always contains the full unfiltered data (from cache or source).
    # `results_df` is what all downstream plot/table cells consume.
    #
    # Metric filtering works on `metric_display` (human-readable names).
    # The Clinician row (completeness-human) is always kept so downstream cells that
    # need the ground-truth score continue to work correctly even when "Clinician" is
    # deselected — exclude it only from display-facing metrics if you deselect it.
    _active_models = set(active_models_widget.value)
    # Intersected with MAIN_DATASETS so a headless run cannot drift back to
    # reporting every loaded dataset when the widget default is not applied.
    _active_datasets = set(active_datasets_widget.value) & set(MAIN_DATASETS)
    _active_metrics = set(active_metrics_widget.value)

    results_df = results_df_all[
        results_df_all['eval_model'].isin(_active_models) &
        results_df_all['dataset'].isin(_active_datasets) &
        (
            results_df_all['metric_display'].isin(_active_metrics) |
            (results_df_all['metric'] == 'completeness-human')  # always keep ground truth
        )
    ].copy()

    print(
        f'Active filter: {len(_active_models)} model(s), {len(_active_datasets)} dataset(s), '
        f'{len(_active_metrics)} metric(s) → {len(results_df):,} rows '
        f'(of {len(results_df_all):,} total).'
    )
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary Statistics
    """)
    return


@app.cell
def _(mo, results_df):
    _counts = (
        results_df
        .groupby(["eval_model", "dataset", "metric_display"])
        .size()
        .rename("count")
        .reset_index()
        .pivot_table(
            index="eval_model",
            columns=["dataset", "metric_display"],
            values="count",
            aggfunc="first",
        )
    ).T

    mo.ui.table(_counts, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All Results
    """)
    return


@app.cell
def _(mo, results_df):
    # Display
    mo.ui.dataframe(results_df, page_size=15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Completeness-only Results
    """)
    return


@app.cell
def _(COMPLETE_MODE, completeness_methods, mo, pd, results_df, typing):
    def create_model_human_comparison_df(
            results_df: pd.DataFrame,
            completeness_methods: dict[str, str],
            mode_model: str = 'strict',
            mode_human: str = 'strict'
    ) -> pd.DataFrame:
        """Compare human and model binary completeness judgments.
        Uses 'completeness-human' score as ground truth instead of dataset-specific proxies.
        Returns a DataFrame with human_judgment and model_judgment columns.
        """
        label_map = {True: 'Complete', False: 'Incomplete'}

        # Build lookup: (eval_model, dataset, id) -> score
        # Note: We use the 'score' from completeness-human rows.
        _df = results_df.reset_index().copy()
        human_scores = _df[_df['metric'] == 'completeness-human'].set_index(['dataset', 'id'])['score'].to_dict()

        def get_human_judgement(row: pd.Series) -> typing.Optional[bool]:
            """Determine human judgment based on completeness-human score with dataset-specific handling."""
            score = human_scores.get((row.dataset, row.id))
            # In strict mode, only 1.0 (5/5, 3/3, etc.) is considered complete.
            if mode_human == "strict":
                return score == 1.0
            # Else, relaxed mode. Thresholds can be dataset-specific; here we use 0.75 as a general relaxed threshold
            return score >= 0.75

        def get_model_judgement(row: pd.Series) -> typing.Optional[bool]:
            """Determine model judgment based on the row's metric and score, with optional strictness."""
            if mode_model == 'strict':
                return row.score == 1.0

            # Relaxed thresholds per metric (FSS variants share the same rubric as their base)
            if row.metric == 'completeness-criteria-checklist-healthbench':
                return row.score >= 0.75  # TODO: calibrate this threshold based on human score distributions for HealthBench
            if row.metric in ('completeness-omission-harm', 'completeness-omission-harm-few-shot', 'completeness-medexpert-prometheus'):
                return row.points >= 4
            if row.metric in ('completeness-clinical-significance', 'completeness-clinical-significance-few-shot', 'completeness-medical-safety-prometheus'):
                return row.points >= 2
            return None

        # Restrict to Completeness methods other than human
        df = _df[_df['metric'].isin(completeness_methods)].copy()
        df = df[df.metric != "completeness-human"]

        df['human_judgment'] = df.apply(get_human_judgement, axis="columns").map(label_map)
        df['model_judgment'] = df.apply(get_model_judgement, axis="columns").map(label_map)
        return df


    completeness_accuracy_df = create_model_human_comparison_df(results_df, completeness_methods, mode_human=COMPLETE_MODE, mode_model=COMPLETE_MODE)
    completeness_accuracy_df = completeness_accuracy_df[completeness_accuracy_df['dataset'] != 'K-QA'].reset_index(drop=True)

    # Tag each row as reported or excluded BEFORE remapping metric keys to display names.
    # The paper reports two completeness definitions on MedExpert; the Zero-shot
    # Omission-Harm, Prometheus and General-Likert variants are appendix-only.
    #
    # This replaces the former in-dataset / cross-dataset split. That distinction only
    # existed to separate a rubric applied to the dataset it was annotated for from one
    # applied to the other dataset, and it drove a second label vocabulary
    # (Agrees/Disagrees) in the qualitative analysis. With both definitions now scored
    # against the same MedExpert clinician annotations, every reported pair is in-dataset
    # and TP/TN/FP/FN applies throughout.
    _MAIN_PAIRS = {
        ("completeness-omission-harm-few-shot", "MedExpert"),
        ("completeness-criteria-checklist-healthbench", "MedExpert"),
    }
    completeness_accuracy_df['pairing_type'] = completeness_accuracy_df.apply(
        lambda row: "in-dataset" if (row['metric'], row['dataset']) in _MAIN_PAIRS else "excluded",
        axis=1,
    )

    completeness_accuracy_df['metric'] = completeness_accuracy_df['metric'].map(
        lambda x: completeness_methods.get(x, x)
    )

    # Display
    mo.ui.dataframe(completeness_accuracy_df, page_size=10)
    return completeness_accuracy_df, create_model_human_comparison_df


@app.cell
def _(COMPLETE_MODE, completeness_methods, create_model_human_comparison_df, results_df_all):
    # Escape hatch: the same comparison frame WITHOUT the MAIN_DATASETS restriction.
    # Only the cells that deliberately span datasets use this — the inter-judge
    # specification table and the HealthBench per-criterion agreement table, both of
    # which feed the parked HealthBench analysis rather than this paper.
    completeness_accuracy_df_all = create_model_human_comparison_df(
        results_df_all, completeness_methods, mode_human=COMPLETE_MODE, mode_model=COMPLETE_MODE
    )
    completeness_accuracy_df_all = completeness_accuracy_df_all[
        completeness_accuracy_df_all['dataset'] != 'K-QA'
    ].reset_index(drop=True)
    completeness_accuracy_df_all['metric'] = completeness_accuracy_df_all['metric'].map(
        lambda x: completeness_methods.get(x, x)
    )
    return (completeness_accuracy_df_all,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Count per Dataset
    """)
    return


@app.cell
def _(OUTPUT_DIR, completeness_accuracy_df, mo, os):
    # Unique examples per dataset (one row per id×dataset)
    _unique = (
        completeness_accuracy_df[['id', 'dataset', 'human_judgment']]
        .drop_duplicates(subset=['id', 'dataset'])
    )
    _counts = (
        _unique.groupby(['dataset', 'human_judgment'])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis='columns')
        [['Complete', 'Incomplete']]
    )
    _counts['Total'] = _counts['Complete'] + _counts['Incomplete']
    _counts['% Incomplete'] = (_counts['Incomplete'] / _counts['Total'] * 100).round(1)
    dataset_counts_df = _counts.reset_index().rename(columns={'dataset': 'Dataset'})

    # Save LaTeX
    _out_path = f'{OUTPUT_DIR}/dataset_counts.tex'
    os.makedirs(os.path.dirname(_out_path) if os.path.dirname(_out_path) else '.', exist_ok=True)
    _tex = dataset_counts_df.to_latex(
        index=False,
        caption='Number of complete and incomplete examples per dataset.',
        label='tab:dataset_counts',
        column_format='lrrrr',
        escape=True,
    )
    with open(_out_path, 'w') as _f:
        _f.write(_tex)
    print(f'Saved {_out_path}')

    mo.ui.table(dataset_counts_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # RQ1: In-Dataset Evaluation — Can LLM Judge Reliably Distinguish Complete from Incomplete?

    **Scope:** Restricted to *in-dataset* pairings only — Omission-Harm Rubric on MedExpert; Criteria-Checklist on HealthBench.
    Prometheus and General-Likert variants are appendix-only and excluded here.
    Cross-dataset pairings (Omission-Harm on HealthBench; Criteria-Checklist on MedExpert) are analyzed in **rq3** below.

    Model scores and clinician scores are each binarised into **Complete / Incomplete** using the threshold defined by `COMPLETE_MODE`:
    - **Strict**: score = 1.0 → Complete (no omissions / all criteria met)
    - **Relaxed**: score ≥ 0.75 → Complete (allows minor omissions)

    The resulting binary labels are compared across all sections below (confusion matrices, IAA, discriminative ability, triage utility).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Score Distributions

    Boxplots compare the score distribution of each grading method and the Clinician side-by-side per dataset.
    Compression (narrower spread) or skew (higher/lower median) relative to the Clinician column indicates systematic bias.
    """)
    return


@app.cell
def _(
    SETTINGS,
    active_metrics_widget,
    completeness_methods,
    pd,
    results_df,
    save_figure,
    sns,
):
    # Non-human completeness methods only — restricted to metrics active in results_df
    # and the active metrics widget selection (GL excluded by default)
    _active_non_human = [
        m for m in results_df['metric'].unique()
        if m in completeness_methods and m != 'completeness-human'
        and completeness_methods[m] in active_metrics_widget.value
    ]
    _non_human_methods = {k: completeness_methods[k] for k in _active_non_human}

    completeness_df = results_df[results_df['metric'].isin(_non_human_methods)].copy()
    completeness_df['metric'] = completeness_df['metric'].map(_non_human_methods)

    # Build Clinician rows: deduplicate per (id, dataset) then replicate across each method column
    _human_rows = (
        results_df[results_df['metric'] == 'completeness-human']
        .reset_index()
        .drop_duplicates(subset=['id', 'dataset'])
        .copy()
    )
    _human_rows['eval_model'] = 'Clinician'
    _clinician_dfs = []
    for _method_display in _non_human_methods.values():
        _copy = _human_rows.copy()
        _copy['metric'] = _method_display
        _clinician_dfs.append(_copy)
    completeness_df = pd.concat([completeness_df] + _clinician_dfs, ignore_index=True)

    _model_order = [s['model'] for s in SETTINGS] + ['Clinician']
    _palette = dict(zip(_model_order, sns.color_palette('tab10', n_colors=len(_model_order))))

    _g = sns.FacetGrid(completeness_df, col='metric', row='dataset', height=3, aspect=1.2, sharey=True)
    _g.map_dataframe(sns.boxplot, x='eval_model', y='score', hue='eval_model', order=_model_order, hue_order=_model_order, palette=_palette, legend=False, width=0.4, flierprops=dict(marker='o', markersize=2, alpha=0.4))
    _g.set_titles('{col_name}')
    _g.set(ylim=(0, 1))
    _g.set_axis_labels('', 'Score')
    for _ax in _g.axes.flat:
        _ax.set_xticks(range(len(_model_order)))
        _ax.set_xticklabels(_model_order, rotation=20, ha='right', rotation_mode='anchor')
    for _ax, _row_name in zip(_g.axes[:, 0], _g.row_names):
        _ax.annotate(_row_name, xy=(-0.2, 0.5), xycoords='axes fraction', ha='right', va='center', rotation=90, size='large', weight='bold')
    save_figure(_g, 'completeness_metric_distribution')
    _g.figure
    return (completeness_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Score Distribution by Clinician Judgment

    Boxplots compare the score distribution of each grading method for clinician-labeled Complete vs.\ Incomplete responses, faceted by dataset (rows) and backbone LLM (columns).
    Well-separated box pairs indicate the model score is a useful discriminator even before binarisation.
    """)
    return


@app.cell
def _(SETTINGS, completeness_df, save_figure, sns, sp):
    _plot_df = completeness_df[
        (completeness_df['eval_model'] != 'Clinician') &
        completeness_df['human_judgment'].notna()
    ].copy()

    _model_order = [s['model'] for s in SETTINGS if s['model'] in _plot_df['eval_model'].unique()]
    _dataset_order = sorted(_plot_df['dataset'].unique())
    _metric_order = list(dict.fromkeys(_plot_df['metric']))
    _judgment_order = ['Complete', 'Incomplete']
    _hue_palette = {'Complete': '#1f77b4', 'Incomplete': '#ff7f0e'}

    _g = sns.FacetGrid(
        _plot_df, row='dataset', col='eval_model',
        col_order=_model_order, row_order=_dataset_order,
        height=2.5, aspect=1.3, sharey=True
    )
    _g.map_dataframe(
        sns.boxplot, x='metric', y='score', hue='human_judgment',
        order=_metric_order, hue_order=_judgment_order,
        palette=_hue_palette, width=0.5,
        flierprops=dict(marker='o', markersize=2, alpha=0.4)
    )

    # Collect significance results and track significant set for plot annotations
    _sig, _nonsig, _sig_set = [], [], set()
    for _row_name in _dataset_order:
        for _col_name in _model_order:
            _sub = _plot_df[(_plot_df['dataset'] == _row_name) & (_plot_df['eval_model'] == _col_name)]
            for _m in _metric_order:
                _m_sub = _sub[_sub['metric'] == _m]
                _comp = _m_sub[_m_sub['human_judgment'] == 'Complete']['score'].dropna()
                _inc = _m_sub[_m_sub['human_judgment'] == 'Incomplete']['score'].dropna()
                _label = f'{_row_name} / {_col_name} / {_m}'
                if len(_comp) < 5 or len(_inc) < 5:
                    _nonsig.append(f'{_label} (insufficient data)')
                else:
                    _, _p = sp.stats.mannwhitneyu(_comp, _inc, alternative='greater')
                    if _p < 0.05:
                        _sig.append(f'{_label} (p={_p:.3f})')
                        _sig_set.add((_row_name, _col_name, _m))
                    else:
                        _nonsig.append(f'{_label} (p={_p:.3f})')
    print('=' * 60)
    print('  BOXPLOT: Complete > Incomplete (Mann-Whitney one-sided)')
    print('=' * 60)
    print(f'  SIGNIFICANT (p < 0.05) — {len(_sig)} pairs:')
    for _s in _sig:
        print(f'    [SIG]  {_s}')
    print(f'  NOT SIGNIFICANT (p >= 0.05) — {len(_nonsig)} pairs:')
    for _ns in _nonsig:
        print(f'    [NS]   {_ns}')
    print('=' * 60)

    # Add hatching for significant pairs — identify boxes by x-center, not index
    for _ri, _row_name in enumerate(_dataset_order):
        for _ci, _col_name in enumerate(_model_order):
            _ax = _g.axes[_ri, _ci]
            for _patch in _ax.patches:
                try:
                    _verts = _patch.get_path().vertices
                    _x_center = _verts[:, 0].mean()
                    _mi_nearest = round(_x_center)
                    if 0 <= _mi_nearest < len(_metric_order):
                        _m = _metric_order[_mi_nearest]
                        if (_row_name, _col_name, _m) in _sig_set:
                            _patch.set_hatch('//')
                            _patch.set_edgecolor('black')
                except Exception:
                    pass

    for _ri, _row_axes in enumerate(_g.axes):
        for _ci, _ax in enumerate(_row_axes):
            _ax.set_title(_model_order[_ci] if _ri == 0 else '')
    _g.set(ylim=(0, 1))
    _g.set_axis_labels('', 'Score')
    for _ax in _g.axes.flat:
        _ax.set_xticks(range(len(_metric_order)))
        _ax.set_xticklabels(_metric_order, rotation=20, ha='right', rotation_mode='anchor')
    for _ax, _row_name in zip(_g.axes[:, 0], _dataset_order):
        _ax.annotate(_row_name, xy=(-0.2, 0.5), xycoords='axes fraction', ha='right', va='center', rotation=90, size='large', weight='bold')
    _g.add_legend(title='Clinician Label', loc='lower center', bbox_to_anchor=(0.45, 1.01), ncol=10, frameon=True)
    save_figure(_g, 'completeness_human_model_boxplot')
    _g.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Omission-Harm Rubric Scores by Clinician Severity (MedExpert)

    Row-normalised heatmap: for each clinician-labelled omission severity level, shows the
    percentage of MedExpert responses assigned each model score (1–5) by the Omission-Harm
    Rubric FS. All five main-paper backbone models, including Prometheus-2 7B, use the same
    few-shot prompt (the Table 1 scores); Prometheus's native rubric run is not pooled in.
    A well-calibrated model concentrates mass in scores 4–5 for None/Mild severity and
    scores 1–3 for Severe/Life-threatening.
    """)
    return


@app.cell
def _(SETTINGS, np, pd, plt, results_df_all, save_figure, sns):
    # Few-shot Omission-Harm only. The Prometheus folders also hold a native-rubric run
    # ('completeness-medexpert-prometheus'); pooling it in mixed two prompts in the
    # Prometheus panel (~50% of scores >= 4 vs 10.8% for the few-shot run in Table 1).
    _METRIC_KEYS = {'completeness-omission-harm-few-shot'}
    _SEVERITY_ORDER = ['No Omissions', 'Mild', 'Moderate', 'Severe', 'Life-threatening']
    _SEVERITY_MAP = {
        None: 'No Omissions',
        'None': 'No Omissions',
        'Mild - no action is required': 'Mild',
        'Moderate - may negatively impact the patients health if no action is taken': 'Moderate',
        'Severe - may require medical intervention by a doctor': 'Severe',
        'Life-threatening - can be life-threatening without medical intervention': 'Life-threatening',
    }

    def _severity_of(meta):
        # A default here would silently file any wording drift in the clinician
        # severity strings under "no omissions", so unmapped values raise instead.
        raw = meta.get('omission_severity') if isinstance(meta, dict) else None
        if raw not in _SEVERITY_MAP:
            raise ValueError(f'unmapped omission_severity: {raw!r}')
        return _SEVERITY_MAP[raw]

    _MAIN_MODELS = ['LLaMA 3.3-70B', 'OpenBioLLM', 'GPT-5 Mini', 'Qwen3.6', 'Prometheus-2 7B']
    _model_order = [s['model'] for s in SETTINGS if s['model'] in _MAIN_MODELS]

    _sub = results_df_all[
        (results_df_all['dataset'] == 'MedExpert') &
        (results_df_all['metric'].isin(_METRIC_KEYS)) &
        (results_df_all['eval_model'].isin(_MAIN_MODELS))
    ].copy()
    if _sub.empty:
        raise ValueError(
            f'no MedExpert rows matched {_METRIC_KEYS} — check the metric keys against '
            'completeness_methods/LEGACY_METRIC_ALIASES before trusting this figure'
        )
    _sub['severity'] = _sub['meta'].apply(_severity_of)
    _sub['severity'] = pd.Categorical(_sub['severity'], categories=_SEVERITY_ORDER, ordered=True)
    _sub['points'] = _sub['points'].astype(int)

    _ncols = 2
    _nrows = (len(_model_order) + 1) // 2
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(7, 3.2 * _nrows), sharey=True)
    _axes_flat = list(_axes.flat)

    # Shared colorbar axis
    _cbar_ax = _fig.add_axes([1.01, 0.15, 0.02, 0.7])

    for _i, _model in enumerate(_model_order):
        _ax = _axes_flat[_i]
        _mdf = _sub[_sub['eval_model'] == _model]
        if _mdf.empty:
            # reindex(fill_value=0) below would turn this into a plausible-looking
            # all-zero grid rather than failing, so say so loudly.
            raise ValueError(f'no rows for backbone {_model!r} in {_METRIC_KEYS}')
        _pivot = _mdf.groupby(['severity', 'points'], observed=True).size().unstack(fill_value=0)
        for _col in range(1, 6):
            if _col not in _pivot.columns:
                _pivot[_col] = 0
        _pivot = _pivot[[c for c in range(1, 6)]].reindex(_SEVERITY_ORDER, fill_value=0)
        _pct = _pivot.div(_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100

        _draw_cbar = (_i == len(_model_order) - 1)
        sns.heatmap(
            _pct, ax=_ax,
            cmap='YlOrRd', vmin=0, vmax=100,
            annot=True, fmt='.0f', annot_kws={'size': 8},
            cbar=_draw_cbar, cbar_ax=(_cbar_ax if _draw_cbar else None),
            linewidths=0.5, linecolor='white',
        )
        if _draw_cbar:
            _cbar_ax.set_ylabel('% of responses at severity level', fontsize=8)
            _cbar_ax.tick_params(labelsize=7)

        _ax.set_title(_model, fontsize=10, weight='bold')
        _ax.set_xlabel('Omission-Harm Rubric score', fontsize=8)
        _ax.set_ylabel('Worst clinician-rated omission' if _i % _ncols == 0 else '', fontsize=8)
        _ax.tick_params(labelsize=8)

    for _j in range(len(_model_order), len(_axes_flat)):
        _axes_flat[_j].set_visible(False)

    _fig.tight_layout()
    save_figure(_fig, 'completeness_omission_severity')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discriminative Ability

    Quantifies how well the model's continuous completeness score separates clinician-labeled Complete vs Incomplete responses.

    | Metric | Interpretation |
    |--------|----------------|
    | **AUC-ROC** | Area under the ROC curve; 0.5 = random, 1.0 = perfect separation |
    | **AUC-PR** | |
    | **F1** | Harmonic mean of precision and recall on the binary Complete label |
    | **Point-Biserial r** | Correlation between continuous score and binary label; sign and magnitude indicate direction and strength |
    | **Mann-Whitney U p-value** | One-sided test: are Complete scores stochastically greater than Incomplete scores? |
    | **IAA (alpha)** | Treats the clinician and the model as two annotators and computes **Krippendorff's α** on their binary Complete/Incomplete labels. |
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    average_precision_score,
    completeness_accuracy_df,
    mo,
    np,
    pd,
    precision_recall_fscore_support,
    roc_auc_score,
    simpledorff,
    sp,
):
    def calculate_discrimination_stats(df: pd.DataFrame) -> pd.Series:
        """AUC, F1, Mann-Whitney U p-value, IAA, and point-biserial r for a metric vs binary ground truth."""
        y_true = df['human_judgment'].values
        y_true_binary = df['human_judgment'].map({'Complete': 1, 'Incomplete': 0}).values
        y_pred = df["model_judgment"].values
        y_pred_prob = df['score'].values

        # ROC-AUC
        try:
            auc = roc_auc_score(y_true_binary, y_pred_prob, average=None)
        except Exception as err:
            print(f"Error in AUC calculation: {err}")
            auc = float("nan")

        # AUC-PR (positive class = Incomplete = 0 in y_true_binary, so flip)
        y_true_incomplete = 1 - y_true_binary  # Incomplete=1, Complete=0
        y_score_incomplete = -y_pred_prob       # higher score = more Complete, so negate
        try:
            auc_pr = average_precision_score(y_true_incomplete, y_score_incomplete)
        except Exception as err:
            print(f"Error in AUC-PR calculation: {err}")
            auc_pr = float("nan")

        # Mann-Whitney U p-value
        complete_scores = y_pred_prob[y_true == "Complete"]
        incomplete_scores = y_pred_prob[y_true == "Incomplete"]
        try:
            _, p_value = sp.stats.mannwhitneyu(complete_scores, incomplete_scores, alternative='greater')
        except Exception as err:
            print(f"Error in Mann-Whitneu U p-value calculation {err}")
            p_value = float("nan")

        # point-biserial r
        try:
            r_value, _ = sp.stats.pointbiserialr(y_true_binary, y_pred_prob)
        except Exception as err:
            print(f"Error in Pearson correlation calculation: {err}")
            r_value = float("nan")

        # F1, precision, and recall
        # NOTE: We care about Incomplete responses, so that is the "positive" class
        group_label = df.name if hasattr(df, 'name') else '?'
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label='Incomplete', zero_division=0, average="binary")
        except Exception as err:
            print(f"Error in F1 calculation [{group_label}]: {err!r}")
            print(f"  y_true unique={pd.Series(y_true).unique()}, y_pred unique={pd.Series(y_pred).unique()}, n={len(y_true)}")
            prec, rec, f1 = float("nan"), float("nan"), float("nan")

        # Raw agreement (accuracy) between clinician and model verdicts
        try:
            acc = float((y_true == y_pred).mean())
        except Exception as err:
            print(f"Error in accuracy calculation [{group_label}]: {err!r}")
            acc = float("nan")

        # Krippendorff's alpha
        melted = df.melt(id_vars='id', value_vars=['human_judgment', 'model_judgment'], var_name='annotator', value_name='annotation')
        try:
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(melted, experiment_col='id', annotator_col='annotator', class_col='annotation')
        except Exception as err:
            print(f"Error in alpha calculation [{group_label}]: {err!r}")
            print(f"  annotation unique={melted['annotation'].unique()}, n={len(melted)}")
            alpha = np.nan

        # Return metrics
        return pd.Series({
            "alpha": alpha,
            'Accuracy': acc,
            'F1': f1,
            "Precision": prec,
            "Recall": rec,
            'AUC': auc,
            'AUC-PR': auc_pr,
            'MW-U p-value': '{:.2e}'.format(p_value),
            'Point-Biserial r': r_value,
            "n_samples": len(df),
        })


    # Calculate scores — restricted to in-dataset pairings (Omission-Harm→MedExpert; Criteria-Checklist→HealthBench).
    # Prometheus and General-Likert are excluded (pairing_type == "excluded").
    # NOTE: Criteria-Checklist on HealthBench is an in-dataset pairing but produces artificially high AUC-PR
    #       due to circular logic (ICL ground truth was derived with the same method). Discussed in paper.
    discrim_results_df = completeness_accuracy_df[
        completeness_accuracy_df['pairing_type'] == 'in-dataset'
    ].groupby(['eval_model', 'dataset', 'metric']).apply(calculate_discrimination_stats, include_groups=False)

    # Sort and save
    discrim_results_df = discrim_results_df.sort_values(
        ['dataset', 'eval_model', 'metric']
    ).reset_index()
    discrim_results_df.to_csv(f'{OUTPUT_DIR}/rq1_summary.csv', index=False)

    # Display with 2 d.p. — round a display copy so the exported df keeps full precision
    mo.ui.table(discrim_results_df.round(2), selection=None, page_size=len(discrim_results_df))
    return calculate_discrimination_stats, discrim_results_df


@app.cell
def _(
    OUTPUT_DIR,
    average_precision_score,
    completeness_accuracy_df,
    discrim_results_df,
    krippendorff,
    mo,
    np,
    os,
    pd,
    precision_recall_fscore_support,
    sp,
):
    def bootstrap_discrimination_ci(df, n_resamples=1000, seed=42):
        """95% BCa bootstrap CIs for F1, Precision, Recall, AUC-PR (Incomplete class).
        Uses scipy.stats.bootstrap with paired=True for joint resampling of label pairs.
        """
        y_true = df['human_judgment'].values
        y_pred = df['model_judgment'].values
        y_score = df['score'].values
        # AUC-PR convention: Incomplete=1, negate score (same as calculate_discrimination_stats)
        y_inc = (y_true == 'Incomplete').astype(int)
        y_score_neg = -y_score

        results = {}

        # F1, Precision, Recall — single bootstrap call returning all three
        def prf_stat(yt, yp):
            p, r, f, _ = precision_recall_fscore_support(yt, yp, pos_label='Incomplete', zero_division=0, average='binary')
            return np.array([f, p, r])

        try:
            res_prf = sp.stats.bootstrap(
                (y_true, y_pred), prf_stat,
                n_resamples=n_resamples, paired=True,
                method='BCa', confidence_level=0.95, rng=seed,
                vectorized=False,
            )
            for i, name in enumerate(['F1', 'Precision', 'Recall']):
                results[f'{name}_lo'] = res_prf.confidence_interval.low[i]
                results[f'{name}_hi'] = res_prf.confidence_interval.high[i]
        except Exception:
            for name in ['F1', 'Precision', 'Recall']:
                results[f'{name}_lo'] = np.nan
                results[f'{name}_hi'] = np.nan

        # AUC-PR
        try:
            res_auc = sp.stats.bootstrap(
                (y_inc, y_score_neg),
                average_precision_score,
                n_resamples=n_resamples,
                paired=True,
                method='BCa',
                confidence_level=0.95,
                rng=seed,
            )
            results['AUC-PR_lo'] = res_auc.confidence_interval.low
            results['AUC-PR_hi'] = res_auc.confidence_interval.high
        except Exception:
            results['AUC-PR_lo'] = np.nan
            results['AUC-PR_hi'] = np.nan

        # Accuracy (raw agreement)
        def acc_stat(yt, yp):
            return float((yt == yp).mean())

        try:
            res_acc = sp.stats.bootstrap(
                (y_true, y_pred), acc_stat,
                n_resamples=n_resamples, paired=True,
                method='BCa', confidence_level=0.95, rng=seed,
                vectorized=False,
            )
            results['Accuracy_lo'] = res_acc.confidence_interval.low
            results['Accuracy_hi'] = res_acc.confidence_interval.high
        except Exception:
            results['Accuracy_lo'] = np.nan
            results['Accuracy_hi'] = np.nan

        # Krippendorff's alpha (nominal) — reliability matrix shape: (2, n_items)
        # Encode Complete=0, Incomplete=1
        _label_map = {'Complete': 0, 'Incomplete': 1}
        _r1 = np.array([_label_map.get(v, np.nan) for v in y_true], dtype=float)
        _r2 = np.array([_label_map.get(v, np.nan) for v in y_pred], dtype=float)

        def _kalpha(r1, r2):
            return krippendorff.alpha(np.vstack([r1, r2]), level_of_measurement='nominal')

        try:
            results['Krippendorff_alpha'] = _kalpha(_r1, _r2)
            res_ka = sp.stats.bootstrap(
                (_r1, _r2), _kalpha,
                n_resamples=n_resamples, paired=True,
                method='BCa', confidence_level=0.95, rng=seed,
                vectorized=False,
            )
            results['Krippendorff_alpha_lo'] = res_ka.confidence_interval.low
            results['Krippendorff_alpha_hi'] = res_ka.confidence_interval.high
        except Exception:
            results['Krippendorff_alpha'] = np.nan
            results['Krippendorff_alpha_lo'] = np.nan
            results['Krippendorff_alpha_hi'] = np.nan

        return pd.Series(results)

    with mo.persistent_cache(name="bootstrap_ci"):
        bootstrap_ci_df = (
            completeness_accuracy_df[
                completeness_accuracy_df['pairing_type'] == 'in-dataset'
            ]
            .groupby(['eval_model', 'dataset', 'metric'])
            .apply(bootstrap_discrimination_ci, include_groups=False)
            .reset_index()
        )
    # Every pairing that has a bootstrap CI keeps it. discrim_results_df is
    # in-dataset only, so merging the CIs onto it alone silently dropped the
    # Both completeness definitions now live in discrim_results_df, so there is no
    # second point-estimate frame to concatenate. The merge stays how='outer' so an
    # unmatched row shows up as a warning instead of disappearing.
    _point_estimates = discrim_results_df
    discrim_full_df = _point_estimates.merge(
        bootstrap_ci_df,
        on=['eval_model', 'dataset', 'metric'],
        how='outer',
        indicator='_merge_source',
    )
    _unmatched = discrim_full_df[discrim_full_df['_merge_source'] != 'both']
    if len(_unmatched):
        print(f'WARNING: {len(_unmatched)} pairing(s) missing a point estimate or a CI:')
        print(_unmatched[['eval_model', 'dataset', 'metric', '_merge_source']].to_string(index=False))
    discrim_full_df = discrim_full_df.drop(columns='_merge_source')
    discrim_full_df.to_csv(os.path.join(OUTPUT_DIR, 'discrimination_full.csv'), index=False)
    return bootstrap_ci_df, discrim_full_df


@app.cell
def _(OUTPUT_DIR, completeness_accuracy_df, mo, os, pd, simpledorff, sp):
    # Do the two completeness definitions agree with *each other*?
    # RQ1 compares each definition to the clinician separately; this compares
    # them to one another, holding the backbone and the response fixed, so it
    # separates "equal aggregate performance" from "same per-item verdicts".
    # completeness_accuracy_df['metric'] carries the *display* names from
    # `completeness_methods`, not the grader keys, so match on those.
    _OH_METRIC = 'Omission-Harm Rubric'
    _CC_METRIC = 'Criteria-Checklist'

    def rubric_pair_agreement(df, metric_a=_OH_METRIC, metric_b=_CC_METRIC, dataset='MedExpert'):
        """Per-judge agreement between two completeness definitions on the same responses.

        Reports raw verdict agreement, Pearson r on the raw 0-1 scores, and
        Krippendorff's alpha on the binary verdicts, treating the two rubrics as
        two annotators of the same item. Incomplete rates are carried alongside
        because raw agreement and alpha diverge sharply when one rubric labels
        nearly everything Incomplete.
        """
        rows = []
        sub = df[(df['dataset'] == dataset) & (df['metric'].isin({metric_a, metric_b}))]
        for judge, group in sub.groupby('eval_model'):
            a = group[group['metric'] == metric_a].dropna(subset=['model_judgment']).set_index('id')
            b = group[group['metric'] == metric_b].dropna(subset=['model_judgment']).set_index('id')
            shared = a.index.intersection(b.index)
            if len(shared) == 0:
                continue
            a, b = a.loc[shared], b.loc[shared]
            melted = pd.DataFrame({
                'id': list(shared) * 2,
                'annotator': [metric_a] * len(shared) + [metric_b] * len(shared),
                'annotation': list(a['model_judgment']) + list(b['model_judgment']),
            })
            try:
                alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                    melted, experiment_col='id', annotator_col='annotator', class_col='annotation'
                )
            except Exception as err:
                print(f'Error in rubric-pair alpha [{judge}]: {err!r}')
                alpha = float('nan')
            try:
                r_value, p_value = sp.stats.pearsonr(a['score'].values, b['score'].values)
            except Exception as err:
                print(f'Error in rubric-pair Pearson r [{judge}]: {err!r}')
                r_value, p_value = float('nan'), float('nan')
            rows.append({
                'eval_model': judge,
                'dataset': dataset,
                'n': len(shared),
                'raw_agreement': float((a['model_judgment'].values == b['model_judgment'].values).mean()),
                'pearson_r': r_value,
                'pearson_p': p_value,
                'alpha': alpha,
                'incomplete_rate_a': float((a['model_judgment'] == 'Incomplete').mean()),
                'incomplete_rate_b': float((b['model_judgment'] == 'Incomplete').mean()),
            })
        return pd.DataFrame(rows).sort_values('eval_model').reset_index(drop=True)

    # Pooling across judges would inflate Pearson r by folding in between-judge
    # mean differences, so this stays per-judge.
    rubric_agreement_df = rubric_pair_agreement(completeness_accuracy_df)
    rubric_agreement_df.to_csv(os.path.join(OUTPUT_DIR, 'rubric_pair_agreement.csv'), index=False)

    mo.ui.table(rubric_agreement_df.round(3), selection=None, page_size=len(rubric_agreement_df))
    return (rubric_agreement_df,)



@app.cell
def _(np, re):
    # MedExpert has 530 responses to 108 questions (each question answered by up to
    # five chatbots), so responses to the same question are not independent. These
    # helpers let a bootstrap resample questions instead of responses. The question
    # key is parsed from meta['title'] (e.g. 'PC-gemma2-q49-iaa' -> 'PC-q49'); the
    # question text cannot be used because two questions have two wordings each.
    _QUESTION_TITLE_RE = re.compile(r'^(MH|PC)-[a-z0-9]+-(q\d+)')

    def question_cluster_id(title):
        """'{domain}-{question}' key for a MedExpert title, or None if it doesn't parse."""
        match = _QUESTION_TITLE_RE.match(str(title))
        return f'{match.group(1)}-{match.group(2)}' if match else None

    def make_cluster_sampler(cluster_ids):
        """Return draw(rng) -> row indices for one cluster-bootstrap resample.

        Clusters are drawn with replacement and every row of a drawn cluster is kept,
        so a resample has the same number of clusters (not rows) as the original.
        """
        _, inverse = np.unique(np.asarray(cluster_ids), return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
        n_clusters = len(members)

        def draw(rng):
            return np.concatenate([members[k] for k in rng.integers(0, n_clusters, n_clusters)])

        return draw

    return make_cluster_sampler, question_cluster_id


@app.cell
def _(OUTPUT_DIR, average_precision_score, completeness_accuracy_df, make_cluster_sampler, mo, np, os, pd):
    # Is the verdict-level difference between the two completeness definitions
    # real? Both definitions score the same responses against the same clinician
    # labels, so the difference is resampled directly (paired) instead of being
    # read off two marginal intervals, which discard the pairing and are several
    # times wider. Equivalence is tested with two one-sided tests (TOST): the
    # 90% interval on the paired difference must fall inside +/- EQUIV_MARGIN_F1.
    EQUIV_MARGIN_F1 = 2.0  # F1 points, scaled by 100

    def _f1_incomplete(y_true_incomplete, y_pred_incomplete):
        """F1 for the Incomplete class from boolean arrays (fast path for the bootstrap)."""
        tp = np.count_nonzero(y_true_incomplete & y_pred_incomplete)
        fp = np.count_nonzero(~y_true_incomplete & y_pred_incomplete)
        fn = np.count_nonzero(y_true_incomplete & ~y_pred_incomplete)
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0

    def _paired_metrics(y_true_incomplete, y_pred_incomplete, scores):
        """(F1, AUC-PR) for the Incomplete class. Scores run high = more complete, so negate."""
        try:
            auc_pr = average_precision_score(y_true_incomplete, -scores)
        except ValueError:
            auc_pr = np.nan
        return _f1_incomplete(y_true_incomplete, y_pred_incomplete), auc_pr

    def paired_definition_bootstrap(
        df,
        metric_a='Omission-Harm Rubric',
        metric_b='Criteria-Checklist',
        dataset='MedExpert',
        n_resamples=10000,
        margin=EQUIV_MARGIN_F1,
        seed=42,
        cluster_col=None,
    ):
        """Paired bootstrap of the metric_b - metric_a difference, per backbone.

        Resamples responses with replacement (or, with `cluster_col`, whole clusters
        such as questions), recomputes both definitions on each
        resample and keeps the difference, so the interval reflects that the two
        definitions saw identical responses and identical clinician labels.
        Returns point estimates, 90% and 95% intervals, a two-sided bootstrap
        p-value, and the TOST equivalence verdict at `margin` F1 points.
        """
        rng = np.random.default_rng(seed)
        rows = []
        sub = df[(df['dataset'] == dataset) & (df['metric'].isin({metric_a, metric_b}))]
        for judge, group in sub.groupby('eval_model'):
            a = group[group['metric'] == metric_a].dropna(subset=['model_judgment', 'human_judgment']).set_index('id')
            b = group[group['metric'] == metric_b].dropna(subset=['model_judgment', 'human_judgment']).set_index('id')
            shared = a.index.intersection(b.index)
            if len(shared) == 0:
                continue
            a, b = a.loc[shared], b.loc[shared]
            if not (a['human_judgment'].values == b['human_judgment'].values).all():
                print(f'WARNING: clinician labels differ across definitions for {judge}; skipping.')
                continue
            y_true = (a['human_judgment'].values == 'Incomplete')
            pred_a = (a['model_judgment'].values == 'Incomplete')
            pred_b = (b['model_judgment'].values == 'Incomplete')
            score_a, score_b = a['score'].values, b['score'].values
            f1_a, auc_a = _paired_metrics(y_true, pred_a, score_a)
            f1_b, auc_b = _paired_metrics(y_true, pred_b, score_b)

            n = len(shared)
            if cluster_col is None:
                draw = lambda _rng: _rng.integers(0, n, n)
            else:
                draw = make_cluster_sampler(a[cluster_col].values)
            deltas = {'F1': [], 'AUC-PR': []}
            for _ in range(n_resamples):
                idx = draw(rng)
                # A resample with one class missing has no defined AUC-PR.
                if y_true[idx].all() or not y_true[idx].any():
                    continue
                boot_a = _paired_metrics(y_true[idx], pred_a[idx], score_a[idx])
                boot_b = _paired_metrics(y_true[idx], pred_b[idx], score_b[idx])
                deltas['F1'].append(boot_b[0] - boot_a[0])
                deltas['AUC-PR'].append(boot_b[1] - boot_a[1])

            for metric_name, point in (('F1', f1_b - f1_a), ('AUC-PR', auc_b - auc_a)):
                draws = np.asarray(deltas[metric_name]) * 100
                lo90, hi90 = np.percentile(draws, [5, 95])
                lo95, hi95 = np.percentile(draws, [2.5, 97.5])
                # Two-sided bootstrap p-value: twice the smaller tail mass at zero.
                p_value = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
                equivalent = bool(lo90 > -margin and hi90 < margin)
                differs = bool(lo95 > 0 or hi95 < 0)
                rows.append({
                    'eval_model': judge,
                    'dataset': dataset,
                    'metric': metric_name,
                    'n': n,
                    f'{metric_name}_a': (f1_a if metric_name == 'F1' else auc_a) * 100,
                    f'{metric_name}_b': (f1_b if metric_name == 'F1' else auc_b) * 100,
                    'delta': point * 100,
                    'ci90_lo': lo90,
                    'ci90_hi': hi90,
                    'ci95_lo': lo95,
                    'ci95_hi': hi95,
                    'p_two_sided': p_value,
                    'margin': margin,
                    'equivalent': equivalent,
                    'differs_from_zero': differs,
                    'tost_verdict': (
                        'equivalent' if equivalent and not differs
                        else 'different' if differs and not equivalent
                        else 'equivalent but non-zero' if equivalent and differs
                        else 'inconclusive'
                    ),
                    'n_resamples_used': len(draws),
                })
        return pd.DataFrame(rows).sort_values(['metric', 'eval_model']).reset_index(drop=True)

    with mo.persistent_cache(name="paired_definition_bootstrap"):
        paired_definition_df = paired_definition_bootstrap(completeness_accuracy_df)
    paired_definition_df.to_csv(os.path.join(OUTPUT_DIR, 'paired_definition_bootstrap.csv'), index=False)

    mo.ui.table(paired_definition_df.round(3), selection=None, page_size=len(paired_definition_df))
    return paired_definition_bootstrap, paired_definition_df


@app.cell
def _(discrim_full_df, mo, pd):
    def _fmt_ci_display(est, lo, hi):
        try:
            est, lo, hi = float(est), float(lo), float(hi)
        except (ValueError, TypeError):
            return '--'
        if any(pd.isna(v) for v in [est, lo, hi]):
            return '--'
        return f'{est:.2f} [{lo:.2f}, {hi:.2f}]'

    _display_df = discrim_full_df.assign(
        **{
            'F1 [95% CI]': discrim_full_df.apply(lambda r: _fmt_ci_display(r['F1'], r['F1_lo'], r['F1_hi']), axis=1),
            'Precision [95% CI]': discrim_full_df.apply(lambda r: _fmt_ci_display(r['Precision'], r['Precision_lo'], r['Precision_hi']), axis=1),
            'Recall [95% CI]': discrim_full_df.apply(lambda r: _fmt_ci_display(r['Recall'], r['Recall_lo'], r['Recall_hi']), axis=1),
            'AUC-PR [95% CI]': discrim_full_df.apply(lambda r: _fmt_ci_display(r['AUC-PR'], r['AUC-PR_lo'], r['AUC-PR_hi']), axis=1),
            'α [95% CI]': discrim_full_df.apply(lambda r: _fmt_ci_display(r['Krippendorff_alpha'], r['Krippendorff_alpha_lo'], r['Krippendorff_alpha_hi']), axis=1),
        }
    )[['eval_model', 'dataset', 'metric', 'α [95% CI]', 'F1 [95% CI]', 'Precision [95% CI]', 'Recall [95% CI]', 'AUC-PR [95% CI]']]
    mo.vstack([
        mo.md("### Discrimination Stats with Confidence Intervals (95% CI)"),
        mo.ui.table(_display_df, selection=None),
    ])
    return


@app.cell
def _(
    DATASETS,
    OUTPUT_DIR,
    SETTINGS,
    completeness_methods,
    discrim_full_df,
    os,
    pd,
):
    _ci_table_path = os.path.join(OUTPUT_DIR, 'discrimination_stats_rq1_ci.tex')
    _dataset_order_ci = [ds for ds in DATASETS if ds in discrim_full_df['dataset'].unique()]
    _metric_display_order_ci = [
        v for v in completeness_methods.values()
        if v != 'Clinician' and v in discrim_full_df['metric'].unique()
    ]
    _model_order_ci = [
        s['model'] for s in SETTINGS
        if s['model'] in discrim_full_df['eval_model'].unique()
    ]

    def _fmt_ci(est, lo, hi, rel_margin=False):
        try:
            est, lo, hi = float(est), float(lo), float(hi)
        except (ValueError, TypeError):
            return '--'
        if any(pd.isna(v) for v in [est, lo, hi]):
            return '--'
        if rel_margin and est != 0:
            margin_pct = (hi - lo) / 2 / abs(est) * 100
            cell_text = f'{est:.2f} ({lo:.2f}, {hi:.2f}; {margin_pct:.1f}\\%)'
            if margin_pct <= 10.0:
                return r'\cellcolor{green!20}' + cell_text
            return cell_text
        return f'{est:.2f} ({lo:.2f}, {hi:.2f})'

    _rows_ci = [
        r'\begin{table*}[!t]',
        r'\centering',
        r'\renewcommand{\arraystretch}{1.3}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lllccccc}',
        r'    \toprule',
        r'    \textbf{Dataset} & \textbf{Metric} & \textbf{Model} & $\boldsymbol{\alpha}$ & \textbf{F1} & \textbf{Prec} & \textbf{Rec} & \textbf{AUC-PR} \\',
        r'    \midrule',
    ]

    for _di_ci, _dataset_ci in enumerate(_dataset_order_ci):
        _metrics_in_ds = [
            m for m in _metric_display_order_ci
            if m in discrim_full_df[discrim_full_df['dataset'] == _dataset_ci]['metric'].values
        ]
        if not _metrics_in_ds:
            continue

        _n_ds_rows = len(_metrics_in_ds) * len(_model_order_ci)

        for _mi_ci, _metric_ci in enumerate(_metrics_in_ds):
            for _model_i, _eval_model in enumerate(_model_order_ci):
                _cells = []

                if _mi_ci == 0 and _model_i == 0:
                    _cells.append(r'\multirow{' + str(_n_ds_rows) + r'}{*}{' + _dataset_ci + r'}')
                else:
                    _cells.append('')

                if _model_i == 0:
                    _metric_tex = (
                        r'\shortstack[l]{Omission-Harm Rubric \\ (Zero-shot)}'
                        if _metric_ci == 'Omission-Harm Rubric (Zero-shot)'
                        else _metric_ci
                    )
                    _cells.append(r'\multirow{' + str(len(_model_order_ci)) + r'}{*}{' + _metric_tex + r'}')
                else:
                    _cells.append('')

                _cells.append(_eval_model)

                _row_ci = discrim_full_df[
                    (discrim_full_df['eval_model'] == _eval_model) &
                    (discrim_full_df['dataset'] == _dataset_ci) &
                    (discrim_full_df['metric'] == _metric_ci)
                ]
                if _row_ci.empty:
                    _cells += ['--', '--', '--', '--', '--']
                else:
                    r = _row_ci.iloc[0]
                    _cells += [
                        _fmt_ci(r.get('Krippendorff_alpha'), r.get('Krippendorff_alpha_lo'), r.get('Krippendorff_alpha_hi')),
                        _fmt_ci(r.get('F1'), r.get('F1_lo'), r.get('F1_hi'), rel_margin=True),
                        _fmt_ci(r.get('Precision'), r.get('Precision_lo'), r.get('Precision_hi'), rel_margin=True),
                        _fmt_ci(r.get('Recall'), r.get('Recall_lo'), r.get('Recall_hi'), rel_margin=True),
                        _fmt_ci(r.get('AUC-PR'), r.get('AUC-PR_lo'), r.get('AUC-PR_hi'), rel_margin=True),
                    ]

                _rows_ci.append('    ' + ' & '.join(_cells) + r' \\')

            if _mi_ci < len(_metrics_in_ds) - 1:
                _rows_ci.append(r'    \cmidrule(l){2-8}')

        if _di_ci < len(_dataset_order_ci) - 1:
            _rows_ci.append(r'    \midrule')

    _rows_ci += [
        r'    \bottomrule',
        r'    \end{tabular}',
        r'}% end resizebox',
        r'\caption{95\% bootstrap confidence intervals (1{,}000 BCa resamples) for all discrimination metrics. Each cell shows point estimate (lower, upper; relative margin), where relative margin $= \frac{(upper - lower)/2}{\text{estimate}} \times 100$, expressed as a percentage; omitted for $\alpha$ (Krippendorff\textquotesingle s $\alpha$, nominal) as it can be negative. Cells with relative margin $\leq 10\%$ are highlighted in \colorbox{green!20}{green}. F1, Precision, Recall, and AUC-PR are for the Incomplete class.}',
        r'\label{tab:discrimination_stats_ci}',
        r'\end{table*}',
    ]

    with open(_ci_table_path, 'w') as _f:
        _f.write('\n'.join(_rows_ci) + '\n')
    print(f'Saved {_ci_table_path}')
    return


@app.cell
def _(
    DATASETS,
    OUTPUT_DIR,
    SETTINGS,
    cluster_ci_df,
    completeness_accuracy_df,
    completeness_methods,
    discrim_results_df,
    os,
    pd,
):
    # Discrimination table for both reported completeness definitions on MedExpert.
    # Point estimates and 95% CIs come from cluster_ci_df (ML4H rebuttal section):
    # percentile bootstrap that resamples the 108 questions, because responses to the
    # same question are not independent.
    # Zero-shot Omission-Harm Rubric is excluded.
    _EXCLUDE_METRICS = {'Omission-Harm Rubric (Zero-shot)'}

    # OpenBioLLM is reported on its own in the appendix (tab:openbiollm_discrimination),
    # so it is excluded from the main-text table rather than starred inside it.
    _OPENBIO = 'OpenBioLLM'
    _all_models = [s['model'] for s in SETTINGS]
    _present_models = set(discrim_results_df['eval_model'])
    _model_order = [m for m in _all_models if m in _present_models and m != _OPENBIO]

    _metric_display_order = [
        v for v in completeness_methods.values()
        if v != 'Clinician' and v not in _EXCLUDE_METRICS
    ]

    # Wrap the rubric names over two lines so the Definition column stops
    # driving the table width (the CI stacks pushed it past \textwidth).
    _METRIC_WRAP = {
        'Omission-Harm Rubric': r'\shortstack[l]{Omission-Harm \\ Rubric}',
        'Criteria-Checklist': r'\shortstack[l]{Criteria- \\ Checklist}',
    }

    def _fmt_stat(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return '--'
        if pd.isna(v):
            return '--'
        _txt = f'{v*100:.1f}'
        return '0.0' if _txt == '-0.0' else _txt

    # Question-clustered 95% percentile CIs, stacked under each point estimate.
    _STATS = ['alpha', 'kappa', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC-PR', 'AUROC']
    _ci_lookup = {
        (_r['eval_model'], _r['metric'], _r['stat']): _r
        for _, _r in cluster_ci_df.iterrows()
    }

    def _fmt_cell(eval_model, metric, stat):
        _ci = _ci_lookup.get((eval_model, metric, stat))
        if _ci is None:
            return '--'
        _est = _fmt_stat(_ci['estimate'])
        if _est == '--':
            return _est
        _lo, _hi = _fmt_stat(_ci['question_lo']), _fmt_stat(_ci['question_hi'])
        if _lo == '--' or _hi == '--':
            _lo = _hi = _est
        return r'\shortstack{' + _est + r' \\ {\scriptsize [' + _lo + ', ' + _hi + r']}}'

    # Build section list: (dataset_label, display_df, is_cross_dataset)
    _rq1_clean = discrim_results_df[~discrim_results_df['metric'].isin(_EXCLUDE_METRICS)]

    # MedExpert only: both completeness definitions against real clinician ground
    # truth. HealthBench rows are deliberately absent — their labels came from
    # GPT-4.1 scoring the ideal answer, so an F1 there measures agreement with
    # GPT-4.1 rather than completeness. HealthBench is reported as inter-judge
    # agreement instead (tab:healthbench_judge_agreement).
    _MAIN_DATASETS = ['MedExpert']

    _sections = []
    for _ds in _MAIN_DATASETS:
        _sub = _rq1_clean[_rq1_clean['dataset'] == _ds]
        if not _sub.empty:
            _sections.append((_ds, _sub, False))

    _out_path = os.path.join(OUTPUT_DIR, 'discrimination_stats_medexpert.tex')
    _rows = [
        r'\begin{table}[!t]',
        r'\centering',
        # \small + tighter column padding keep the table inside \textwidth
        # without \resizebox, which would scale the type off the document grid.
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\renewcommand{\arraystretch}{1.1}',
        r'\begin{tabular}{llcccccccc}',
        r'    \toprule',
        r'    \textbf{Definition} & \textbf{Judge} & $\boldsymbol{\alpha}$ & $\boldsymbol{\kappa}$ & \textbf{Acc} & \textbf{F1} & \textbf{Prec} & \textbf{Rec} & \textbf{AUC-PR} & \textbf{AUROC} \\',
        r'    \midrule',
    ]

    for _si, (_ds_label, _sub_df, _is_cross) in enumerate(_sections):
        _metrics_in_ds = [
            m for m in _metric_display_order
            if m in _sub_df['metric'].values
        ]
        if not _metrics_in_ds:
            continue

        _n_ds_rows = len(_metrics_in_ds) * len(_model_order)

        for _mi, _metric in enumerate(_metrics_in_ds):
            for _model_i, _eval_model in enumerate(_model_order):
                # One dataset only, so the dataset column is dropped; _ds_label and
                # _n_ds_rows are retained for the section bookkeeping below.
                _cells = []

                if _model_i == 0:
                    _metric_tex = _METRIC_WRAP.get(_metric, _metric)
                    _cells.append(r'\multirow{' + str(len(_model_order)) + r'}{*}{' + _metric_tex + r'}')
                else:
                    _cells.append('')

                _cells.append(_eval_model)

                _row = _sub_df[
                    (_sub_df['eval_model'] == _eval_model) &
                    (_sub_df['metric'] == _metric)
                ]
                if _row.empty:
                    _cells += ['--'] * len(_STATS)
                else:
                    _cells += [_fmt_cell(_eval_model, _metric, s) for s in _STATS]

                _rows.append('    ' + ' & '.join(_cells) + r' \\')

            if _mi < len(_metrics_in_ds) - 1:
                _rows.append(r'    \cmidrule(l){2-10}')

        # Midrule between sections (not after the last). Both sections are MedExpert
        # now, so there is no cross-dataset banner.
        if _si < len(_sections) - 1:
            _rows.append(r'    \midrule')

    _rows += [
        r'    \bottomrule',
        r'    \end{tabular}',
        r'\caption{Discriminative power of both completeness definitions on MedExpert against '
        r'clinician judgment; values $\times 100$. $\alpha$ = '
        r'Krippendorff\textquotesingle s $\alpha$ (nominal); $\kappa$ = Cohen\textquotesingle s $\kappa$; '
        r'Acc = raw verdict agreement; F1, Precision, Recall, AUC-PR and AUROC are for '
        r'the Incomplete class (random baselines: AUC-PR 34, AUROC 50). Brackets: 95\% '
        r'question-clustered bootstrap confidence intervals (\Cref{app:stats}). '
        r'OpenBioLLM-70B and the zero-shot Omission-Harm '
        r'Rubric are in \Cref{tab:discrimination_stats_full}.}\label{tab:discrimination_stats}',
        r'\end{table}',
    ]

    with open(_out_path, 'w') as _f:
        _f.write('\n'.join(_rows) + '\n')

    # ── Appendix companion: every backbone and prompt configuration ──────────
    # Same columns as tab:discrimination_stats, plus the share of responses each Judge
    # flags as incomplete, for all five backbones (OpenBioLLM included) and the
    # zero-shot Omission-Harm Rubric. The directive comparison (app:directive) and the
    # OpenBioLLM subsection (app:openbiollm) both read off this table.
    _FULL_METRICS = ['Criteria-Checklist', 'Omission-Harm Rubric', 'Omission-Harm Rubric (Zero-shot)']
    _FULL_WRAP = {
        **_METRIC_WRAP,
        'Omission-Harm Rubric (Zero-shot)': r'\shortstack[l]{Omission-Harm \\ Rubric \\ (zero-shot)}',
    }
    _full_models = [m for m in _all_models if m in _present_models]
    _me = completeness_accuracy_df[completeness_accuracy_df['dataset'] == 'MedExpert']
    _pct_flagged = (
        _me.assign(_flag=_me['model_judgment'] == 'Incomplete')
        .groupby(['metric', 'eval_model'])['_flag'].mean()
    )
    _prevalence = (_me.drop_duplicates('id')['human_judgment'] == 'Incomplete').mean()

    _full_path = os.path.join(OUTPUT_DIR, 'discrimination_stats_full.tex')
    _full_rows = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\renewcommand{\arraystretch}{1.1}',
        r'\begin{tabular}{llccccccccc}',
        r'    \toprule',
        r'    \textbf{Definition} & \textbf{Judge} & \textbf{\% Flag} & $\boldsymbol{\alpha}$ & $\boldsymbol{\kappa}$ & \textbf{Acc} & \textbf{F1} & \textbf{Prec} & \textbf{Rec} & \textbf{AUC-PR} & \textbf{AUROC} \\',
        r'    \midrule',
    ]
    for _mi, _metric in enumerate(_FULL_METRICS):
        for _model_i, _eval_model in enumerate(_full_models):
            _cells = [
                r'\multirow{' + str(len(_full_models)) + r'}{*}{' + _FULL_WRAP[_metric] + r'}' if _model_i == 0 else '',
                _eval_model,
            ]
            if (_eval_model, _metric, 'F1') not in _ci_lookup:
                _cells += ['--'] * (len(_STATS) + 1)
            else:
                _cells.append(f'{100 * _pct_flagged[(_metric, _eval_model)]:.1f}')
                _cells += [_fmt_cell(_eval_model, _metric, _s) for _s in _STATS]
            _full_rows.append('    ' + ' & '.join(_cells) + r' \\')
        if _mi < len(_FULL_METRICS) - 1:
            _full_rows.append(r'    \midrule')
    _full_rows += [
        r'    \bottomrule',
        r'    \end{tabular}',
        r'\caption{Verdict-level performance of every backbone under every prompt configuration on '
        r'MedExpert, against clinician judgment: the two definitions in \Cref{tab:discrimination_stats}, '
        r'with OpenBioLLM-70B added, and the zero-shot Omission-Harm Rubric, which drops the few-shot '
        r'examples, the reasoning scaffold and the directive to search for missing information. '
        r'\% Flag = share of responses the Judge rated incomplete (clinician: '
        + f'{100 * _prevalence:.0f}' + r'\%). Other columns and the question-clustered 95\% '
        r'confidence intervals match \Cref{tab:discrimination_stats}; all values are scaled by 100.'
        r'}\label{tab:discrimination_stats_full}',
        r'\end{table}',
    ]
    with open(_full_path, 'w') as _f:
        _f.write('\n'.join(_full_rows) + '\n')
    print(f'Saved {_out_path}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Threshold Sensitivity Analysis

    Assesses how sensitive RQ1 discrimination metrics (F1, AUC-PR) are to the choice of
    human-score binarization threshold. The model judgment threshold is held fixed at relaxed
    defaults; only the ground-truth label boundary varies.

    - **HealthBench**: sweeps `human_score ≥ {0.50, 0.60, 0.70, 0.75, 0.80, 0.90}`
    - **MedExpert**: sweeps four cumulative severity cutpoints —
      *None only* (strict), *None+Mild* (**baseline**), *None+Mild+Moderate*, *None+Mild+Moderate+Severe*
    """)
    return


@app.cell
def _(
    calculate_discrimination_stats,
    completeness_accuracy_df,
    pd,
    results_df,
):
    # ── Attach raw human scores (not in completeness_accuracy_df) ────────────
    _human_df = (
        results_df.reset_index()
        .pipe(lambda d: d[d['metric'] == 'completeness-human'])
        .drop_duplicates(subset=['dataset', 'id'])[['dataset', 'id', 'score']]
        .rename(columns={'score': 'human_score'})
    )
    _base = completeness_accuracy_df.merge(_human_df, on=['dataset', 'id'], how='left')
    _base = _base.dropna(subset=['human_score', 'model_judgment'])

    # ── Shared sweep helper ───────────────────────────────────────────────────
    def sweep_thresholds(base, thresholds):
        """Sweep human binarization thresholds.

        Args:
            base: DataFrame with model_judgment, human_score, eval_model, metric, id columns.
            thresholds: list of (threshold_name, threshold_value) pairs.
        Returns:
            (sensitivity_df, prevalence_df)
        """
        sens_parts, prev_rows = [], []
        _unique_ids = base[['id', 'human_score']].drop_duplicates('id')
        for tname, tval in thresholds:
            sub = base.copy()
            sub['human_judgment'] = (sub['human_score'] >= tval).map(
                {True: 'Complete', False: 'Incomplete'}
            )
            stats = (
                sub.groupby(['eval_model', 'metric'])
                .apply(calculate_discrimination_stats, include_groups=False)
                .reset_index()
                .assign(threshold_name=tname, threshold=tval)
            )
            sens_parts.append(stats)
            labels = _unique_ids['human_score'] >= tval
            n_incomp = int((~labels).sum())
            pct = round(n_incomp / len(labels) * 100, 1) if len(labels) > 0 else float('nan')
            prev_rows.append({
                'threshold_name': tname,
                'threshold': tval,
                'n_complete': int(labels.sum()),
                'n_incomplete': n_incomp,
                'pct_incomplete': pct,
                'random_auc_pr_baseline': round(pct / 100, 3),
            })
        return pd.concat(sens_parts, ignore_index=True), pd.DataFrame(prev_rows)

    # ── MedExpert: cumulative severity sweep ──────────────────────────────────
    _me_thresholds = [
        ('none',                      1.00),
        ('none_mild',                 0.75),
        ('none_mild_moderate',        0.50),
        ('none_mild_moderate_severe', 0.25),
    ]
    me_sensitivity_df, me_prevalence_df = sweep_thresholds(
        _base[_base['dataset'] == 'MedExpert'], _me_thresholds
    )
    return me_prevalence_df, me_sensitivity_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Prevalence by threshold**
    """)
    return


@app.cell
def _(me_prevalence_df, mo):
    me_prevalence_df.insert(0, "dataset", "MedExpert")
    mo.ui.table(me_prevalence_df.round(2), selection=None, page_size=len(me_prevalence_df))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Threshold Sensitivity**
    """)
    return


@app.cell
def _(
    active_metrics_widget,
    active_models_widget,
    me_prevalence_df,
    me_sensitivity_df,
    mlines,
    mo,
    mpatches,
    plt,
    save_figure,
    sns,
):
    def plot_sensitivity(
        sensitivity_df, prevalence_df,
        active_models, active_metrics,
        x_col, x_order, x_labels,
        baseline_x, baseline_label,
        title, filename,
        figsize_per_col=(4.0, 7),
    ):
        """Plot F1 and AUC-PR sensitivity curves across thresholds.

        For numeric axes (HealthBench) x_labels=None and x_order is the sorted value list.
        For categorical axes (MedExpert) x_labels is a dict mapping name → tick string.
        baseline_x is the x-value (numeric) or index (categorical) for the reference vline.
        """
        models = [m for m in active_models if m in sensitivity_df['eval_model'].unique()]
        metrics = [m for m in active_metrics if m in sensitivity_df['metric'].unique()]
        if not models or not metrics:
            return None
        palette = sns.color_palette('tab10', n_colors=len(metrics))
        categorical = x_labels is not None
        x_pos = {v: i for i, v in enumerate(x_order)} if categorical else None
        n_cols = len(models)

        fig, axes = plt.subplots(
            2, n_cols,
            figsize=(figsize_per_col[0] * n_cols, figsize_per_col[1]),
            sharex=True,
        )
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        legend_handles = {}
        for ci, metric in enumerate(metrics):
            color = palette[ci]
            for ri, stat in enumerate(['F1', 'AUC-PR']):
                for mi, model in enumerate(models):
                    ax = axes[ri, mi]
                    sub = sensitivity_df[
                        (sensitivity_df['eval_model'] == model) &
                        (sensitivity_df['metric'] == metric)
                    ]
                    if sub.empty:
                        continue
                    if categorical:
                        xs = [x_pos[v] for v in x_order]
                        ys = [
                            float(sub.loc[sub[x_col] == v, stat].iloc[0])
                            if not sub.loc[sub[x_col] == v].empty else float('nan')
                            for v in x_order
                        ]
                    else:
                        sub = sub.sort_values(x_col)
                        xs, ys = sub[x_col].tolist(), sub[stat].tolist()
                    ax.plot(xs, ys, marker='o', markersize=4, linewidth=1.5, color=color)
                    if metric not in legend_handles:
                        legend_handles[metric] = mlines.Line2D(
                            [], [], color=color, marker='o', markersize=4, linewidth=1.5
                        )

        # AUC-PR row: random baseline ribbon
        prev = prevalence_df.copy()
        for mi in range(n_cols):
            ax = axes[1, mi]
            if categorical:
                bxs = [x_pos[v] for v in x_order]
                bys = [
                    float(prev.loc[prev[x_col] == v, 'random_auc_pr_baseline'].iloc[0])
                    for v in x_order
                ]
            else:
                prev = prev.sort_values(x_col)
                bxs, bys = prev[x_col].tolist(), prev['random_auc_pr_baseline'].tolist()
            ax.fill_between(bxs, 0, bys, alpha=0.15, color='grey', zorder=0)

        # Axis formatting
        for ri, (stat, ylabel) in enumerate(zip(['F1', 'AUC-PR'], ['F1 (Incomplete)', 'AUC-PR (Incomplete)'])):
            for mi, model in enumerate(models):
                ax = axes[ri, mi]
                ax.axvline(baseline_x, color='black', linestyle='--', linewidth=0.8, alpha=0.55)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel(ylabel if mi == 0 else '', fontsize=12)
                ax.tick_params(labelsize=10)
                if mi > 0:
                    ax.tick_params(labelleft=False)
                if ri == 0:
                    ax.set_title(model, fontsize=12, fontweight='bold')
                if ri == 1:
                    ax.set_xlabel('Threshold', fontsize=12)
                if categorical:
                    ax.set_xticks(list(range(len(x_order))))
                    ax.set_xticklabels(
                        [x_labels[v] for v in x_order], rotation=0, ha='center', fontsize=8
                    )
                    ax.set_xlim(-0.3, len(x_order) - 0.7)
                else:
                    ax.set_xlim(min(x_order) - 0.02, max(x_order) + 0.02)

        legend_handles[baseline_label] = mlines.Line2D(
            [], [], color='black', linestyle='--', linewidth=0.8, alpha=0.7
        )
        legend_handles['Random AUC-PR baseline'] = mpatches.Patch(facecolor='grey', alpha=0.4)
        fig.legend(
            list(legend_handles.values()), list(legend_handles.keys()),
            loc='lower center', bbox_to_anchor=(0.5, -0.06),
            ncol=len(legend_handles), fontsize=11, framealpha=0.8, handlelength=1.5,
        )
        fig.suptitle(title, fontsize=20, y=1.01)
        fig.tight_layout()
        save_figure(fig, filename)
        return fig

    _active_models = active_models_widget.value
    _active_metrics = active_metrics_widget.value

    _fig_me = plot_sensitivity(
        me_sensitivity_df, me_prevalence_df,
        active_models=_active_models, active_metrics=_active_metrics,
        x_col='threshold_name',
        x_order=['none', 'none_mild', 'none_mild_moderate', 'none_mild_moderate_severe'],
        x_labels={
            'none':                      'None',
            'none_mild':                 'None+Mild\n(baseline)',
            'none_mild_moderate':        'None+Mild\n+Moderate',
            'none_mild_moderate_severe': 'None+Mild\n+Moderate+Severe',
        },
        baseline_x=1, baseline_label='Baseline (None+Mild)',
        title='MedExpert: Severity Threshold Sensitivity of Discrimination Metrics',
        filename='threshold_sensitivity_medexpert',
        figsize_per_col=(4.5, 8),
    )
    _fig_me
    return


@app.cell
def _(OUTPUT_DIR, me_sensitivity_df, mo, os, pd):
    # MedExpert: three named severity cutpoints (fixed — use actual threshold_name values)
    _me_tab = me_sensitivity_df[
        me_sensitivity_df['threshold_name'].isin(['none', 'none_mild', 'none_mild_moderate'])
    ].assign(
        dataset='MedExpert',
        Threshold=lambda d: d['threshold_name'].map({
            'none':               'Strict (1.0)',
            'none_mild':          r'None+Mild (0.75)$^\dagger$',
            'none_mild_moderate': 'None+Mild+Moderate (0.5)',
        }),
    )

    _pivot = (
        _me_tab[['dataset', 'Threshold', 'eval_model', 'metric', 'F1', 'AUC-PR']]
        .pivot_table(
            index=['dataset', 'Threshold', 'metric'],
            columns='eval_model',
            values=['F1', 'AUC-PR'],
            aggfunc='first',
        )
    )
    _pivot.columns = [f'{em} ({stat})' for stat, em in _pivot.columns]
    _pivot = _pivot.reset_index().round(3)

    _pivot.to_csv(os.path.join(OUTPUT_DIR, 'threshold_sensitivity.csv'), index=False)
    _pivot.to_latex(
        os.path.join(OUTPUT_DIR, 'threshold_sensitivity.tex'),
        index=False, float_format='%.3f', escape=False,
        caption=(
            r'Threshold sensitivity: F1 and AUC-PR under alternative human binarization thresholds. $\dagger$ = baseline used in main results.'
        ),
        label='tab:threshold_sensitivity',
    )
    mo.ui.table(_pivot, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrices

    Binary confusion matrices (clinician as ground truth, model as prediction), normalised by row.
    Each cell shows raw count and row-normalised percentage.
    The top-left cell (Complete → Complete) and bottom-right (Incomplete → Incomplete) are true positives/negatives.
    High off-diagonal mass indicates systematic over- or under-prediction of completeness.
    """)
    return


@app.cell(hide_code=True)
def _(completeness_methods, confusion_matrix, np, sns):
    CM_LABELS = ['Complete', 'Incomplete']

    # Derived from completeness_methods — never goes stale
    PROMETHEUS_DISPLAY_NAMES = frozenset(
        v for k, v in completeness_methods.items() if 'prometheus' in k
    )

    def draw_cm_heatmap(data, **kwargs):
        """FacetGrid map_dataframe callback: draw a row-normalised confusion matrix."""
        try:
            _counts = confusion_matrix(
                data['human_judgment'], data['model_judgment'], labels=CM_LABELS
            )
            _norm = confusion_matrix(
                data['human_judgment'], data['model_judgment'],
                labels=CM_LABELS, normalize='true',
            )
            _annot = np.array(
                [f'{c} ({p:.1%})' for c, p in zip(_counts.flatten(), _norm.flatten())]
            ).reshape(_counts.shape)
        except ValueError:
            _norm = np.zeros((len(CM_LABELS), len(CM_LABELS)))
            _annot = np.full((len(CM_LABELS), len(CM_LABELS)), 'N/A')
        sns.heatmap(
            _norm, annot=_annot, fmt='', cmap='Blues',
            xticklabels=CM_LABELS, yticklabels=CM_LABELS,
            cbar=False, vmin=0, vmax=1,
        )

    return CM_LABELS, draw_cm_heatmap


@app.cell
def _(
    CM_LABELS,
    completeness_accuracy_df,
    confusion_matrix,
    mo,
    np,
    plt,
    save_figure,
    sns,
):
    # RQ1: in-dataset pairings only (Prometheus and General-Likert excluded via pairing_type)
    _filtered_df = completeness_accuracy_df[
        completeness_accuracy_df['pairing_type'] == 'in-dataset'
    ].dropna(subset=['human_judgment', 'model_judgment'])

    # ── Aggregate: one 2×2 per dataset ───────────────────────────────────────
    _datasets = sorted(_filtered_df['dataset'].unique())
    _fig_agg, _axes = plt.subplots(1, len(_datasets), figsize=(4 * len(_datasets), 4))
    if len(_datasets) == 1:
        _axes = [_axes]
    for _ax, _ds in zip(_axes, _datasets):
        _sub = _filtered_df[_filtered_df['dataset'] == _ds]
        _counts = confusion_matrix(_sub['human_judgment'], _sub['model_judgment'], labels=CM_LABELS)
        _norm   = confusion_matrix(_sub['human_judgment'], _sub['model_judgment'], labels=CM_LABELS, normalize='true')
        _annot  = np.array(
            [f'{c}\n({p:.1%})' for c, p in zip(_counts.flatten(), _norm.flatten())]
        ).reshape(_counts.shape)
        sns.heatmap(
            _norm, annot=_annot, fmt='', cmap='Blues',
            xticklabels=CM_LABELS, yticklabels=CM_LABELS,
            cbar=False, vmin=0, vmax=1, ax=_ax,
        )
        _ax.set_title(_ds)
        _ax.set_xlabel('Model Judgment (Predicted)')
        _ax.set_ylabel('Human Judgment (Actual)')
    _fig_agg.suptitle('All Models & Rubrics (Aggregated)', y=1.02)
    _fig_agg.tight_layout()
    save_figure(_fig_agg, 'confusion_matrix_aggregate')
    plt.close('all')
    mo.vstack([mo.md('#### Aggregate Confusion Matrix'), _fig_agg])
    return


@app.cell
def _(
    SETTINGS,
    completeness_accuracy_df,
    draw_cm_heatmap,
    mo,
    plt,
    save_figure,
    sns,
):
    _figs_per_model = []
    for _eval_model_name in [s['model'] for s in SETTINGS]:
        # RQ1: in-dataset pairings only
        _plot_df = completeness_accuracy_df[
            (completeness_accuracy_df['eval_model'] == _eval_model_name) &
            (completeness_accuracy_df['pairing_type'] == 'in-dataset')
        ].reset_index(drop=True)
        if _plot_df.empty:
            continue
        _g = sns.FacetGrid(_plot_df, col='metric', row='dataset', sharex=False, sharey=False, height=4)
        _g.map_dataframe(draw_cm_heatmap)
        _g.set_titles('{row_name} | {col_name}')
        _g.set_axis_labels('Model Judgment (Predicted)', 'Human Judgment (Actual)')
        _g.figure.suptitle(_eval_model_name, y=1.02)
        save_figure(_g, f'confusion_matrix_{_eval_model_name.replace(" ", "_").replace("/", "-")}')
        _figs_per_model.append(_g.figure)
    plt.close('all')
    mo.vstack([mo.md('#### Per-Model Confusion Matrices'), *_figs_per_model])
    return


@app.cell
def _(
    SETTINGS,
    completeness_accuracy_df,
    draw_cm_heatmap,
    mo,
    plt,
    save_figure,
    sns,
):
    # RQ1: in-dataset pairings only (Prometheus and General-Likert excluded via pairing_type)
    _filtered_df2 = completeness_accuracy_df[
        completeness_accuracy_df['pairing_type'] == 'in-dataset'
    ].dropna(subset=['human_judgment', 'model_judgment'])

    # ── Per-rubric: one FacetGrid per rubric, columns = models ───────────────
    _model_order = [s['model'] for s in SETTINGS]
    _rubric_figs = []
    for _rubric in sorted(_filtered_df2['metric'].unique()):
        _rub_df = _filtered_df2[_filtered_df2['metric'] == _rubric]
        _models_present = [m for m in _model_order if m in _rub_df['eval_model'].unique()]
        if not _models_present:
            continue
        _g = sns.FacetGrid(
            _rub_df[_rub_df['eval_model'].isin(_models_present)],
            col='eval_model', col_order=_models_present,
            row='dataset', sharex=False, sharey=False, height=3.5,
        )
        _g.map_dataframe(draw_cm_heatmap)
        _g.set_titles('{row_name} | {col_name}')
        _g.set_axis_labels('Model Judgment (Predicted)', 'Human Judgment (Actual)')
        _g.figure.suptitle(_rubric, y=1.02)
        _safe = _rubric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
        save_figure(_g, f'confusion_matrix_rubric_{_safe}')
        _rubric_figs.append(_g.figure)
        plt.close('all')
    mo.vstack([mo.md('#### Per-Rubric Confusion Matrices'), *_rubric_figs])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inter-LLM Score Agreement

    Pairwise Pearson correlation between eval model scores on the **same responses**, within each dataset × metric. High correlation means models rank responses similarly regardless of which LLM does the grading; low correlation suggests the grading method is sensitive to the choice of eval model.
    """)
    return


@app.cell
def _(
    SETTINGS,
    completeness_accuracy_df,
    itertools,
    mo,
    np,
    pd,
    plt,
    save_figure,
    sns,
    sp,
):
    # Reported pairings only; Zero-shot/Prometheus/General-Likert are 'excluded'.
    _plot_df = completeness_accuracy_df[
        completeness_accuracy_df['pairing_type'] == 'in-dataset'
    ]
    _pivot = _plot_df.pivot_table(
        index=['dataset', 'metric', 'id'],
        columns='eval_model',
        values='score',
        aggfunc='first',
    )
    _datasets = sorted(_plot_df['dataset'].unique())
    _metrics = sorted(_plot_df['metric'].unique())
    # Respect SETTINGS order; only include models actually present
    _all_models = [s['model'] for s in SETTINGS if s['model'] in _plot_df['eval_model'].unique()]

    _cmap = plt.cm.coolwarm.copy()
    _cmap.set_bad('black')

    # All (metric, dataset) panels side by side in one row, so the figure
    # spans the page width instead of running down a column.
    _panels = [(_metric, _dataset) for _metric in _metrics for _dataset in _datasets]
    _fig, _axes = plt.subplots(
        1, len(_panels),
        figsize=(4 * len(_panels), 4),
        squeeze=False,
        gridspec_kw={'wspace': 0.05},
    )

    _summary_rows = []
    for _ci, (_metric, _dataset) in enumerate(_panels):
        _ax = _axes[0][_ci]
        try:
            _sub = _pivot.loc[(_dataset, _metric)].dropna(axis=1, how='all')
            _corr = _sub.corr(method='pearson').reindex(index=_all_models, columns=_all_models)
            _n = len(_sub)
        except KeyError:
            _sub = pd.DataFrame(dtype=float)
            _corr = pd.DataFrame(np.nan, index=_all_models, columns=_all_models)
            _n = 0

        # Single pass over unique pairs: build annot matrix + summary rows
        _annot = pd.DataFrame('', index=_all_models, columns=_all_models)
        for _ma, _mb in itertools.combinations(_all_models, 2):
            _r = _corr.loc[_ma, _mb] if (_ma in _corr.index and _mb in _corr.columns) else np.nan
            if pd.isna(_r):
                continue
            _pair = _sub[[_ma, _mb]].dropna() if (_ma in _sub.columns and _mb in _sub.columns) else pd.DataFrame()
            try:
                _, _p = sp.stats.pearsonr(_pair.iloc[:, 0], _pair.iloc[:, 1])
                _sig = '*' if _p < 0.05 else ''
            except Exception:
                _p, _sig = np.nan, ''
            _label = f'{_r:.2f}{_sig}'
            _annot.loc[_ma, _mb] = _label   # fill both triangles
            _annot.loc[_mb, _ma] = _label
            _summary_rows.append({
                'dataset': _dataset, 'metric': _metric,
                'model_a': _ma, 'model_b': _mb,
                'r': round(_r, 3),
                'p': round(float(_p), 4) if not pd.isna(_p) else np.nan,
                'n': len(_pair),
            })

        sns.heatmap(
            _corr, annot=_annot.to_numpy(), fmt='', cmap=_cmap,
            vmin=-1, vmax=1, square=True, cbar=False,
            linewidths=0.5, linecolor='white', ax=_ax,
        )
        _ax.set_title(f'{_dataset} | {_metric}\n(n={_n})', fontsize=8)
        _ax.set_xlabel('')
        _ax.set_ylabel('')
        _ax.tick_params(axis='x', rotation=25, labelsize=7)
        if _ci == 0:
            _ax.tick_params(axis='y', rotation=0, labelsize=7)
        else:
            _ax.set_yticklabels([])
            _ax.tick_params(axis='y', left=False)

    _fig.tight_layout()
    _fig.suptitle('Inter-Model Score Correlation per LLM and Dataset', y=1.02)
    save_figure(_fig, 'inter_model_score_correlation')
    inter_model_corr_df = pd.DataFrame(_summary_rows)
    mo.vstack([
        _fig,
        mo.ui.table(inter_model_corr_df.round(3), page_size=10, selection=None)
    ])
    return (inter_model_corr_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inter-Model Agreement Summary

    Mean Pearson $r$ across datasets, one heatmap per rubric. Answers: which rubrics produce stable scores regardless of backbone LLM choice?
    """)
    return


@app.cell
def _(SETTINGS, inter_model_corr_df, mo, np, pd, plt, save_figure, sns):
    _short = {
        'LLaMA 3.3-70B': 'LLaMA', 'OpenBioLLM': 'BioLLM',
        'GPT-5 Mini': 'GPT', 'Qwen3.6': 'Qwen',
        'Prometheus-2 7B': 'Prom.', 'Prometheus-2 BGb 8x7B': 'Prom.BGb',
    }
    _all_models = [
        _short.get(s['model'], s['model'])
        for s in SETTINGS
        if s['model'] in inter_model_corr_df['model_a'].unique() or s['model'] in inter_model_corr_df['model_b'].unique()
    ]
    _metrics = sorted(inter_model_corr_df['metric'].unique())

    _cmap = plt.cm.coolwarm.copy()
    _cmap.set_bad('black')

    _fig, _axes = plt.subplots(
        1, len(_metrics),
        figsize=(4 * len(_metrics), 3.5),
        squeeze=False,
        gridspec_kw={'wspace': 0.05},
    )

    for _ci, _metric in enumerate(_metrics):
        _ax = _axes[0][_ci]
        _sub = inter_model_corr_df[inter_model_corr_df['metric'] == _metric]

        # Mean r across datasets for each model pair
        _mean_r = _sub.groupby(['model_a', 'model_b'])['r'].mean().reset_index()

        # Build full symmetric matrix using short names
        _corr = pd.DataFrame(np.nan, index=_all_models, columns=_all_models)
        _annot = pd.DataFrame('', index=_all_models, columns=_all_models)
        for _, _row in _mean_r.iterrows():
            _ma = _short.get(_row['model_a'], _row['model_a'])
            _mb = _short.get(_row['model_b'], _row['model_b'])
            if _ma in _corr.index and _mb in _corr.columns:
                _corr.loc[_ma, _mb] = _row['r']
                _corr.loc[_mb, _ma] = _row['r']
                _annot.loc[_ma, _mb] = f"{_row['r']:.2f}"
                _annot.loc[_mb, _ma] = f"{_row['r']:.2f}"

        sns.heatmap(
            _corr, annot=_annot.to_numpy(), fmt='', cmap=_cmap,
            vmin=-1, vmax=1, square=True, cbar=False,
            linewidths=0.5, linecolor='white', ax=_ax,
        )
        _ax.set_title(f'{_metric}\n(mean $r$ across datasets)', fontsize=8)
        _ax.set_xlabel('')
        _ax.set_ylabel('')
        _ax.tick_params(axis='x', rotation=25, labelsize=7)
        if _ci == 0:
            _ax.tick_params(axis='y', rotation=0, labelsize=7)
        else:
            _ax.set_yticklabels([])
            _ax.tick_params(axis='y', left=False)

    _fig.tight_layout()
    _fig.suptitle('Inter-Model Agreement Summary (mean $r$ across datasets)', y=0.95)
    save_figure(_fig, 'inter_model_agreement_summary')
    mo.vstack([_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inter-Judge Agreement as a Function of Criterion Specification

    Chance-corrected N-rater agreement across all backbone LLM judges, replacing the pairwise Pearson
    correlations above for the paper. Two levels are reported:

    - **Score level** — Krippendorff's $\alpha$ (ordinal) over each judge's response-level score, for every
      dataset $\times$ rubric. This isolates the effect of *how specified the target is*: clinician-written
      criteria are supplied (Criteria-Checklist on HealthBench), no criteria are supplied (Omission-Harm),
      or each judge must generate its own criteria first (Criteria-Checklist on MedExpert).
    - **Criterion level** — agreement on individual criteria for Criteria-Checklist on HealthBench, where
      every judge scores the same clinician-written criteria. GPT-4.1 enters as one judge among equals, not
      as a reference: no accuracy claim is made here.

    Binary criteria will agree more than an ordinal scale simply because the task is easier; this is not
    evidence of quality.
    """)
    return


@app.cell
def _(krippendorff, np):
    def kalpha_n(matrix, level_of_measurement='nominal'):
        """Krippendorff's alpha over N raters.

        `matrix` is raters x units with np.nan for missing judgments, matching the
        `krippendorff` package's reliability_data layout. Kept separate from the
        two-rater `_kalpha` used by the bootstrap CI, which expects a 2-row stack.
        """
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            return np.nan
        # Drop units judged by fewer than two raters — alpha is undefined for them.
        keep = (~np.isnan(data)).sum(axis=0) >= 2
        if keep.sum() < 2:
            return np.nan
        try:
            return krippendorff.alpha(
                reliability_data=data[:, keep],
                level_of_measurement=level_of_measurement,
            )
        except (ValueError, ZeroDivisionError):
            return np.nan

    return (kalpha_n,)


@app.cell
def _(DATASETS, SETTINGS, completeness_accuracy_df_all, kalpha_n, mo, pd):
    # Score-level inter-judge agreement per dataset x rubric, annotated with how
    # specified the judge's target is. This is the quantitative backbone of the
    # "agreement tracks specification" argument in the RQ2 HealthBench subsection.
    _SPEC = {
        ('HealthBench', 'Criteria-Checklist'): 'Clinician criteria supplied',
        ('HealthBench-MT', 'Criteria-Checklist'): 'Clinician criteria supplied',
        ('MedExpert', 'Omission-Harm Rubric'): 'No criteria',
        ('HealthBench', 'Omission-Harm Rubric'): 'No criteria',
        ('HealthBench-MT', 'Omission-Harm Rubric'): 'No criteria',
        ('MedExpert', 'Criteria-Checklist'): 'Judge-generated criteria',
    }
    _judge_order = [s['model'] for s in SETTINGS if s['model'] != 'Prometheus-2 BGb 8x7B']

    _rows_spec = []
    for _ds in DATASETS:
        for _rubric in ['Criteria-Checklist', 'Omission-Harm Rubric']:
            _sub = completeness_accuracy_df_all[
                (completeness_accuracy_df_all['dataset'] == _ds) &
                (completeness_accuracy_df_all['metric'] == _rubric) &
                (completeness_accuracy_df_all['eval_model'].isin(_judge_order))
            ]
            if _sub.empty:
                continue
            _piv = _sub.pivot_table(index='id', columns='eval_model', values='score')
            _cols = [_c for _c in _judge_order if _c in _piv.columns]
            _piv = _piv[_cols].dropna()
            if len(_piv) < 10:
                continue
            _rows_spec.append({
                'dataset': _ds,
                'metric': _rubric,
                'specification': _SPEC.get((_ds, _rubric), '--'),
                'n_judges': len(_cols),
                'n_responses': len(_piv),
                'alpha_ordinal': kalpha_n(_piv.to_numpy().T, 'ordinal'),
            })

    judge_specification_df = pd.DataFrame(_rows_spec)
    mo.ui.table(judge_specification_df.round(3), selection=None, page_size=12)
    return (judge_specification_df,)


@app.cell
def _(OUTPUT_DIR, judge_specification_df, os, pd):
    # LaTeX: score-level inter-judge agreement ordered by how specified the target is.
    _SPEC_ORDER = ['Clinician criteria supplied', 'No criteria', 'Judge-generated criteria']
    _df = judge_specification_df.copy()
    _df['_spec_rank'] = _df['specification'].apply(
        lambda s: _SPEC_ORDER.index(s) if s in _SPEC_ORDER else len(_SPEC_ORDER)
    )
    _df = _df.sort_values(['_spec_rank', 'alpha_ordinal'], ascending=[True, False])

    _path = os.path.join(OUTPUT_DIR, 'judge_agreement_by_specification.tex')
    _rows = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{llcc}',
        r'    \toprule',
        r'    \textbf{Judge target} & \textbf{Rubric $\times$ Dataset} & \textbf{$n$} & $\boldsymbol{\alpha}$ \\',
        r'    \midrule',
    ]
    _prev_spec = None
    for _, _r in _df.iterrows():
        _spec = _r['specification'] if _r['specification'] != _prev_spec else ''
        if _prev_spec is not None and _r['specification'] != _prev_spec:
            _rows.append(r'    \midrule')
        _prev_spec = _r['specification']
        _alpha = '--' if pd.isna(_r['alpha_ordinal']) else f"{_r['alpha_ordinal']:.2f}"
        _rows.append(
            f"    {_spec} & {_r['metric']} $\\times$ {_r['dataset']} "
            f"& {int(_r['n_responses'])} & {_alpha} \\\\"
        )
    _rows += [
        r'    \bottomrule',
        r'\end{tabular}',
        r'\caption{Inter-judge agreement on response-level completeness scores, grouped by how '
        r'specified the judge\textquotesingle s target is. $\alpha$ = Krippendorff\textquotesingle s '
        r'$\alpha$ (ordinal) across backbone LLM judges on the same responses; it measures whether '
        r'judges agree with \emph{each other}, not with clinicians. Agreement is high only when the '
        r'clinician\textquotesingle s criteria are supplied, and collapses when each judge must decide '
        r'for itself what a complete response requires.}\label{tab:judge_agreement_specification}',
        r'\end{table}',
    ]
    with open(_path, 'w') as _f:
        _f.write('\n'.join(_rows) + '\n')
    print(f'Saved {_path}')
    return


@app.cell
def _(json, os, pd, results_df_all):
    # Per-criterion long frame for Criteria-Checklist on HealthBench: one row per
    # (record, criterion index) with each judge's binary met/not-met label plus the
    # GPT-4.1 label from the augmented dataset, treated as one more judge.
    _CC_JUDGES = ['LLaMA 3.3-70B', 'OpenBioLLM', 'GPT-5 Mini', 'Qwen3.6']
    _GPT41 = 'GPT-4.1'
    _HB_PATH = os.path.join(
        os.environ['PROJECT_DIR'], 'data/HealthBench/healthbench_dataset_augmented_w_scores.jsonl'
    )

    _gpt41_criteria = {}
    if os.path.exists(_HB_PATH):
        with open(_HB_PATH) as _f:
            for _line in _f:
                _rec = json.loads(_line)
                _details = (_rec.get('meta') or {}).get('human_completeness_score_details')
                if _details:
                    _gpt41_criteria[_rec['id']] = _details

    _sub = results_df_all.reset_index()
    _sub = _sub[
        (_sub['metric_display'] == 'Criteria-Checklist') &
        (_sub['dataset'].isin(['HealthBench', 'HealthBench-MT'])) &
        (_sub['eval_model'].isin(_CC_JUDGES))
    ]

    _rows_crit = []
    for (_rid, _ds), _g in _sub.groupby(['id', 'dataset']):
        _per_judge = {}
        for _, _row in _g.iterrows():
            _md = _row['metric_details']
            if isinstance(_md, dict) and isinstance(_md.get('meta'), list):
                _per_judge[_row['eval_model']] = _md['meta']
        if len(_per_judge) < 2:
            continue
        _n_crit = min(len(_v) for _v in _per_judge.values())
        if _n_crit == 0:
            continue
        _human = _gpt41_criteria.get(_rid)
        _first = _per_judge[next(iter(_per_judge))]
        for _ci in range(_n_crit):
            _rec_out = {
                'id': _rid,
                'dataset': _ds,
                'criterion_index': _ci,
                'criterion': _first[_ci].get('criterion', ''),
            }
            for _judge, _crit_list in _per_judge.items():
                _score = _crit_list[_ci].get('criteria_score')
                _rec_out[_judge] = None if _score is None else bool(_score)
            if _human and _ci < len(_human):
                _rec_out[_GPT41] = bool(_human[_ci].get('criteria_score', False))
            _rows_crit.append(_rec_out)

    criterion_votes_df = pd.DataFrame(_rows_crit)
    CC_JUDGES = _CC_JUDGES
    GPT41_KEY = _GPT41
    print(
        f'Per-criterion frame: {len(criterion_votes_df)} criterion judgments, '
        f'{criterion_votes_df["id"].nunique() if len(criterion_votes_df) else 0} responses'
    )
    return CC_JUDGES, GPT41_KEY, criterion_votes_df


@app.cell
def _(CC_JUDGES, GPT41_KEY, criterion_votes_df, itertools, kalpha_n, mo, np, pd):
    # Criterion-level and verdict-level agreement among judges on HealthBench.
    if criterion_votes_df.empty:
        criterion_agreement_df = pd.DataFrame()
        criterion_pairwise_df = pd.DataFrame()
        _crit_out = mo.md('⚠️ No per-criterion data — skipping criterion agreement.')
    else:
        _summary, _pairwise = [], []
        for _ds, _d in criterion_votes_df.groupby('dataset'):
            _cols = [_c for _c in CC_JUDGES if _c in _d.columns]
            _M = _d[_cols].astype(float).to_numpy().T
            _complete = ~np.isnan(_M).any(axis=0)
            _vals = _M[:, _complete]
            _unanimous = (_vals == _vals[0]).all(axis=0).mean() if _vals.size else np.nan

            # Verdict level: same judges, same responses, aggregated to the 0.75 threshold.
            _resp = _d.groupby('id')[_cols].mean()
            _verdict = (_resp.dropna() >= 0.75).astype(float).to_numpy().T

            _row = {
                'dataset': _ds,
                'n_judges': len(_cols),
                'n_criteria': int(_complete.sum()),
                'n_responses': int(_resp.dropna().shape[0]),
                'met_rate': float(np.nanmean(_vals)) if _vals.size else np.nan,
                'contested_rate': 1 - _unanimous,
                'alpha_criterion': kalpha_n(_vals),
                'alpha_verdict': kalpha_n(_verdict),
            }
            if GPT41_KEY in _d.columns:
                _cols_h = _cols + [GPT41_KEY]
                _Mh = _d[_cols_h].astype(float).to_numpy().T
                _row['alpha_criterion_with_gpt41'] = kalpha_n(_Mh[:, ~np.isnan(_Mh).any(axis=0)])
            _summary.append(_row)

            for _a, _b in itertools.combinations(range(len(_cols)), 2):
                _pairwise.append({
                    'dataset': _ds,
                    'model_a': _cols[_a],
                    'model_b': _cols[_b],
                    'raw_agreement': float((_vals[_a] == _vals[_b]).mean()) if _vals.size else np.nan,
                    'alpha': kalpha_n(_vals[[_a, _b]]),
                })

        criterion_agreement_df = pd.DataFrame(_summary)
        criterion_pairwise_df = pd.DataFrame(_pairwise)
        _crit_out = mo.vstack([
            mo.ui.table(criterion_agreement_df.round(3), selection=None, page_size=6),
            mo.ui.table(criterion_pairwise_df.round(3), selection=None, page_size=14),
        ])
    _crit_out
    return criterion_agreement_df, criterion_pairwise_df


@app.cell
def _(
    OUTPUT_DIR,
    criterion_agreement_df,
    criterion_pairwise_df,
    os,
    pd,
):
    # LaTeX: HealthBench judge-agreement table. Replaces the HealthBench F1 /
    # precision / recall / AUC-PR rows, which measured agreement with GPT-4.1
    # silver labels rather than completeness.
    if criterion_agreement_df.empty:
        print('No criterion agreement data — skipping healthbench_judge_agreement.tex')
    else:
        _SHORT = {
            'LLaMA 3.3-70B': 'LLaMA', 'OpenBioLLM': 'OpenBio',
            'GPT-5 Mini': 'GPT-5 Mini', 'Qwen3.6': 'Qwen3.6',
        }
        _fmt = lambda v, s=1: '--' if pd.isna(v) else f'{v * (100 if s == 1 else 1):.{1 if s == 1 else 2}f}'
        _datasets = list(criterion_agreement_df['dataset'])

        _path = os.path.join(OUTPUT_DIR, 'healthbench_judge_agreement.tex')
        _rows = [
            r'\begin{table}[t]',
            r'\centering',
            r'\small',
            r'\begin{tabular}{l' + 'c' * len(_datasets) + r'}',
            r'    \toprule',
            r'    & ' + ' & '.join(r'\textbf{' + _d + r'}' for _d in _datasets) + r' \\',
            r'    \midrule',
            r'    \multicolumn{' + str(len(_datasets) + 1) + r'}{l}{\textit{Criterion level}} \\',
        ]

        def _line(label, key, pct=True):
            _cells = []
            for _d in _datasets:
                _v = criterion_agreement_df.loc[
                    criterion_agreement_df['dataset'] == _d, key
                ]
                _val = _v.iloc[0] if len(_v) else float('nan')
                if pd.isna(_val):
                    _cells.append('--')
                elif key.startswith('n_'):
                    _cells.append(f'{int(_val)}')
                elif pct:
                    _cells.append(f'{_val * 100:.1f}')
                else:
                    _cells.append(f'{_val:.2f}')
            return f'    {label} & ' + ' & '.join(_cells) + r' \\'

        _rows.append(_line('Criteria judged', 'n_criteria'))
        _rows.append(_line('Criteria met (\\%)', 'met_rate'))
        _rows.append(_line('Contested by $\\geq\\!1$ judge (\\%)', 'contested_rate'))
        _rows.append(_line(r'Krippendorff\textquotesingle s $\alpha$', 'alpha_criterion', pct=False))
        if 'alpha_criterion_with_gpt41' in criterion_agreement_df.columns:
            _rows.append(_line(r'\quad including GPT-4.1 as a judge', 'alpha_criterion_with_gpt41', pct=False))

        _rows.append(r'    \midrule')
        _rows.append(r'    \multicolumn{' + str(len(_datasets) + 1) + r'}{l}{\textit{Response verdict level}} \\')
        _rows.append(_line('Responses', 'n_responses'))
        _rows.append(_line(r'Krippendorff\textquotesingle s $\alpha$', 'alpha_verdict', pct=False))

        if not criterion_pairwise_df.empty:
            _rows.append(r'    \midrule')
            _rows.append(r'    \multicolumn{' + str(len(_datasets) + 1) + r'}{l}{\textit{Pairwise criterion agreement, \% raw ($\alpha$)}} \\')
            _pairs = criterion_pairwise_df[['model_a', 'model_b']].drop_duplicates().values.tolist()
            for _ma, _mb in _pairs:
                _cells = []
                for _d in _datasets:
                    _sel = criterion_pairwise_df[
                        (criterion_pairwise_df['dataset'] == _d) &
                        (criterion_pairwise_df['model_a'] == _ma) &
                        (criterion_pairwise_df['model_b'] == _mb)
                    ]
                    if _sel.empty:
                        _cells.append('--')
                    else:
                        _r = _sel.iloc[0]
                        _cells.append(f"{_r['raw_agreement'] * 100:.1f} ({_r['alpha']:.2f})")
                _label = f"{_SHORT.get(_ma, _ma)} vs.\\ {_SHORT.get(_mb, _mb)}"
                _rows.append(f'    {_label} & ' + ' & '.join(_cells) + r' \\')

        _rows += [
            r'    \bottomrule',
            r'\end{tabular}',
            r'\caption{Agreement among LLM Judges on HealthBench, where every judge scores the same '
            r'clinician-written criteria. No accuracy is reported: GPT-4.1 is included as one judge '
            r'among equals rather than as ground truth, so nothing here is a claim about completeness. '
            r'Criterion-level and verdict-level agreement are comparable, and both are inflated by the '
            r'binary format relative to an ordinal scale.}\label{tab:healthbench_judge_agreement}',
            r'\end{table}',
        ]
        with open(_path, 'w') as _f:
            _f.write('\n'.join(_rows) + '\n')
        print(f'Saved {_path}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # RQ2: Beyond verdict agreement, how well do LLM Judge and clinician reasoning align — and what failure patterns arise when verdicts diverge?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Qualitative Analysis Spreadsheet (`rq2_explanation_alignment.xlsx`)

    The spreadsheet contains **four sheets**, one per confusion-matrix category:

    | Sheet | Human Judgment | Model Judgment | Interpretation |
    |-------|---------------|----------------|----------------|
    | **TP** | Incomplete | Incomplete | Both agree: response is incomplete |
    | **TN** | Complete | Complete | Both agree: response is complete |
    | **FP** | Complete | Incomplete | Model falsely flags as incomplete |
    | **FN** | Incomplete | Complete | Model misses an incomplete response |

    Each sheet pools all eval models, datasets, and metrics.
    Rows are sampled up to *N* per (eval\_model × dataset × metric) group to balance representation, then shuffled.
    The **Eval Model** column is hidden by default; unhide to filter by backbone LLM.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_qualitative_examples_widget = mo.ui.number(
        start=1, stop=50, step=1, value=5,
        label="**Examples per group (eval_model × dataset × metric)**",
    )
    mo.output.replace(n_qualitative_examples_widget)
    return (n_qualitative_examples_widget,)


@app.cell
def _(
    OUTPUT_DIR,
    completeness_accuracy_df,
    mo,
    n_qualitative_examples_widget,
    os,
    pd,
    results_df,
):
    _n_examples = n_qualitative_examples_widget.value

    # ── Build lookup: (dataset, id) → completeness-human row (for human score) ──
    _human_rows = results_df.reset_index()[results_df.reset_index()['metric'] == 'completeness-human']
    _human_rows = _human_rows.drop_duplicates(subset=['dataset', 'id'])
    _human_lookup = _human_rows.set_index(['dataset', 'id'])[['score', 'points', 'total_points', 'metric_details']].to_dict('index')

    # ── Load HealthBench human scoring details (criteria + GPT-4.1 explanations) ──
    _hb_path = os.path.join(os.environ['PROJECT_DIR'], 'data', 'HealthBench', 'healthbench_dataset_augmented_w_scores.jsonl')
    _hb_details: dict = {}
    if os.path.exists(_hb_path):
        _hb_df = pd.read_json(_hb_path, lines=True)
        for _, _hb_row in _hb_df.iterrows():
            _details = (_hb_row.get('meta') or {}).get('human_completeness_score_details')
            if _details:
                _hb_details[str(_hb_row['id'])] = _details

    # ── Restrict to the reported pairings ──
    # 'excluded' rows (Zero-shot Omission-Harm, General-Likert, Prometheus-specific
    # metrics) are appendix-only and already absent via pairing_type assignment.
    _all_df = completeness_accuracy_df[
        completeness_accuracy_df['pairing_type'] == 'in-dataset'
    ].copy()

    # Positive class = Incomplete. Both completeness definitions are scored against the
    # same MedExpert clinician annotations, so the accuracy framing applies to both;
    # the former Agrees/Disagrees vocabulary for cross-dataset pairings is retired.
    _CATEGORY_BY_VERDICT = {
        ('Incomplete', 'Incomplete'): 'TP',
        ('Complete',   'Complete'):   'TN',
        ('Complete',   'Incomplete'): 'FP',
        ('Incomplete', 'Complete'):   'FN',
    }

    def _get_category(row):
        return _CATEGORY_BY_VERDICT.get((row['human_judgment'], row['model_judgment']), 'Unknown')

    _all_df['_category'] = _all_df.apply(_get_category, axis=1)

    # ── Formatting helpers ──
    def _fmt_human_judgment(row):
        h = _human_lookup.get((row['dataset'], row['id']), {})
        score = h.get('score')
        score_str = f'{score:.2f}' if score is not None else '?'
        return f"{row['human_judgment']} ({score_str})"

    def _fmt_model_judgment(row):
        pts, tot = row.get('points'), row.get('total_points')
        if pts is not None and tot is not None:
            return f"{row['model_judgment']} ({int(pts)}/{int(tot)})"
        return f"{row['model_judgment']} ({row['score']:.2f})"

    def _fmt_human_explanation(row):
        if row['dataset'] == 'MedExpert':
            omissions = row.get('omissions') or []
            meta = row.get('meta') or {}
            overall_severity = meta.get('omission_severity')
            if not omissions:
                return 'No omissions identified.'
            parts = []
            for om in omissions:
                if isinstance(om, dict):
                    text = (om.get('omission_text') or om.get('text') or
                            om.get('description') or om.get('omission') or str(om))
                    sev = (om.get('omission_severity') or om.get('severity') or
                           om.get('harm_severity') or 'unknown severity')
                    parts.append(f'- {text} ({sev})')
                else:
                    parts.append(f'- {om}')
            h = _human_lookup.get((row['dataset'], row['id']), {})
            score = h.get('score')
            header_lines = []
            if score is not None:
                header_lines.append(f'Clinician score: {score:.2f}')
            if overall_severity:
                header_lines.append(f'Omission severity: {overall_severity}')
            header = '\n'.join(header_lines) + '\n\n' if header_lines else ''
            return header + '\n'.join(parts)
        elif row['dataset'].startswith('HealthBench'):
            criteria = _hb_details.get(str(row['id']), [])
            if not criteria:
                # Fallback to results_df metric_details if JSONL not available
                h = _human_lookup.get((row['dataset'], row['id']), {})
                criteria = (h.get('metric_details') or {}).get('meta', [])
            present = [c for c in criteria if isinstance(c, dict) and c.get('criteria_score')]
            absent  = [c for c in criteria if isinstance(c, dict) and not c.get('criteria_score')]
            parts = []
            if present:
                lines = [f'PRESENT ({len(present)}/{len(criteria)} criteria):']
                for c in present:
                    lines.append(f"  \u2713 [{c.get('points', '')} pts] {c.get('criterion', '')}")
                    if c.get('explanation'):
                        lines.append(f"    \u2192 {c['explanation']}")
                parts.append('\n'.join(lines))
            if absent:
                lines = [f'NOT PRESENT ({len(absent)}/{len(criteria)} criteria):']
                for c in absent:
                    lines.append(f"  \u2717 [{c.get('points', '')} pts] {c.get('criterion', '')}")
                    if c.get('explanation'):
                        lines.append(f"    \u2192 {c['explanation']}")
                parts.append('\n'.join(lines))
            return '\n\n'.join(parts) if parts else 'No criteria data.'
        return ''

    _likert_metrics = {'Omission-Harm Rubric (Zero-shot)', 'Omission-Harm Rubric', 'General-Likert', 'General-Likert (FS)',
                       'Omission-Harm Rubric (Prometheus)', 'General-Likert (Prometheus)'}

    def _fmt_model_explanation(row):
        criteria = (row.get('metric_details') or {}).get('meta', [])
        if row['metric'] in _likert_metrics:
            parts = []
            for c in criteria:
                if not isinstance(c, dict):
                    continue
                parts.append(f"Score: {c.get('criteria_score')}\n\n{c.get('explanation', '')}")
            return '\n\n---\n\n'.join(parts)
        else:  # Criteria-Checklist
            present = [c.get('criterion', '') for c in criteria if isinstance(c, dict) and c.get('criteria_score')]
            absent  = [c.get('criterion', '') for c in criteria if isinstance(c, dict) and not c.get('criteria_score')]
            parts = []
            if present:
                parts.append('PRESENT:\n' + '\n'.join(f'  - {c}' for c in present))
            if absent:
                parts.append('NOT PRESENT:\n' + '\n'.join(f'  - {c}' for c in absent))
            return '\n\n'.join(parts)

    # ── Sample up to _n_examples per (category, eval_model, dataset, metric) ──
    _sampled = (
        _all_df
        .groupby(['_category', 'eval_model', 'dataset', 'metric'], group_keys=False)
        .apply(lambda g: g.sample(n=min(_n_examples, len(g)), random_state=42))
    )

    # ── Helper to build formatted output dataframe from any dataframe ──
    def _build_out_df(source_df):
        df = source_df.reset_index(drop=True)

        # Vectorize Human Judgment: "Judgment (score)"
        _key_tuples = list(zip(df['dataset'], df['id']))
        _hl_scores = [_human_lookup.get(k, {}).get('score') for k in _key_tuples]
        _score_strs = pd.array([f'{s:.2f}' if s is not None else '?' for s in _hl_scores])
        _human_judgment = df['human_judgment'] + ' (' + _score_strs + ')'

        # Vectorize Model Judgment: "Judgment (pts/tot)" or "Judgment (score)"
        _has_pts = df['points'].notna() & df['total_points'].notna()
        _pts_str = df['points'].where(_has_pts).fillna(0).astype(int).astype(str)
        _tot_str = df['total_points'].where(_has_pts).fillna(0).astype(int).astype(str)
        _score_fmt = df['score'].map('{:.2f}'.format)
        _model_judgment = (
            (df['model_judgment'] + ' (' + _pts_str + '/' + _tot_str + ')')
            .where(_has_pts, df['model_judgment'] + ' (' + _score_fmt + ')')
        )

        return pd.DataFrame({
            '_category':                     df['_category'],
            'Dataset':                       df['dataset'],
            'ID':                            df['id'],
            'Metric':                        df['metric'],
            'Eval Model':                    df['eval_model'],
            'Question':                      df['question'],
            'Answer':                        df['answer'],
            'Human Judgment':                _human_judgment,
            'Model Judgment':                _model_judgment,
            'Explanation of Human Judgment': df.apply(_fmt_human_explanation, axis=1),
            'Explanation of Model Judgment': df.apply(_fmt_model_explanation, axis=1),
            'Notes':                         '',
            'Omissions Aligned?':            '',
        })

    # ── Build formatted output rows (sampled) ──
    _out_df = _build_out_df(_sampled)

    # ── Build full (unsampled) formatted output rows ──
    full_qual_df = _build_out_df(_all_df)

    # ── Excel column config (excludes internal _category column) ──
    _sheet_cols  = [c for c in _out_df.columns if c != '_category']
    _col_widths  = {
        'Dataset': 12, 'ID': 22, 'Metric': 22, 'Eval Model': 22, 'Pairing Type': 16,
        'Question': 55, 'Answer': 65,
        'Human Judgment': 24, 'Model Judgment': 24,
        'Explanation of Human Judgment': 65,
        'Explanation of Model Judgment': 65,
        'Notes': 30,
        'Omissions Aligned?': 20
    }
    _wrap_cols   = {'Question', 'Answer', 'Explanation of Human Judgment',
                    'Explanation of Model Judgment', 'Notes', 'Omissions Aligned?'}
    _hidden_cols = {'Eval Model'}

    _qual_dir = os.path.join(OUTPUT_DIR, 'qualitative_analysis')
    os.makedirs(_qual_dir, exist_ok=True)
    _output_path = os.path.join(_qual_dir, 'rq2_explanation_alignment.xlsx')
    _sampled_parts = []
    with pd.ExcelWriter(_output_path, engine='xlsxwriter') as _writer:
        for _cat in ['TP', 'TN', 'FP', 'FN']:
            _sheet_df = (
                _out_df[_out_df['_category'] == _cat][_sheet_cols]
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            _sheet_df.to_excel(_writer, sheet_name=_cat, index=False)
            _wb = _writer.book
            _ws = _writer.sheets[_cat]
            _wrap_fmt  = _wb.add_format({'text_wrap': True, 'valign': 'top'})
            _plain_fmt = _wb.add_format({'valign': 'top'})
            _ws.freeze_panes(1, 0)
            for _ci, _col in enumerate(_sheet_cols):
                _ws.set_column(
                    _ci, _ci,
                    _col_widths.get(_col, 20),
                    _wrap_fmt if _col in _wrap_cols else _plain_fmt,
                    {'hidden': True} if _col in _hidden_cols else {},
                )
            # Collect the shuffled rows (with _category) for the sampled JSONL
            _sampled_parts.append(
                _out_df[_out_df['_category'] == _cat]
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

    # Write sampled JSONL (same rows and order as the Excel sheets)
    _sampled_path = os.path.join(_qual_dir, 'rq2_qualitative_sampled.jsonl')
    _sampled_export_df = (
        pd.concat(_sampled_parts, ignore_index=True)
        .rename(columns={'_category': 'analysis_category'})
    )
    _sampled_export_df.to_json(_sampled_path, orient='records', lines=True, force_ascii=False)

    _all_cats = ['TP', 'TN', 'FP', 'FN']
    _counts = _out_df.groupby('_category').size().reindex(_all_cats, fill_value=0)
    mo.vstack([
        mo.md('## RQ2 Qualitative Analysis Spreadsheet'),
        mo.md('Accuracy framing against the clinician annotation: TP / TN / FP / FN'),
        mo.md('\n'.join(f'- **{cat}**: {_counts[cat]} rows' for cat in _all_cats)),
        mo.md(f'Excel saved to `{_output_path}`'),
        mo.md(f'Sampled JSONL saved to `{_sampled_path}`'),
    ])
    qual_sheet_cols = _sheet_cols
    qual_xlsx_df = _out_df[_sheet_cols].reset_index(drop=True)
    sampled_qual_df = _sampled_export_df
    return full_qual_df, qual_xlsx_df, sampled_qual_df


@app.cell
def _(mo, qual_xlsx_df):
    mo.ui.dataframe(qual_xlsx_df, page_size=15)
    return


@app.cell
def _(OUTPUT_DIR, full_qual_df, mo, os):
    _rq2_out_dir = os.path.join(OUTPUT_DIR, 'qualitative_analysis')
    _path = os.path.join(_rq2_out_dir, 'rq2_qualitative.jsonl')
    _export_df = full_qual_df.rename(columns={'_category': 'analysis_category'})
    _export_df.to_json(_path, orient='records', lines=True, force_ascii=False)
    _all_cats = ['TP', 'TN', 'FP', 'FN']
    _counts = _export_df.groupby('analysis_category').size().reindex(_all_cats, fill_value=0)

    mo.vstack([
        mo.md('## RQ2 Qualitative JSONL Export'),
        mo.md('Accuracy framing against the clinician annotation: TP / TN / FP / FN'),
        mo.md('\n'.join(f'- **{cat}**: {_counts[cat]} rows' for cat in _all_cats)),
        mo.md(f'Saved to `{_path}`'),
    ])
    return


@app.cell
def _(
    OUTPUT_DIR,
    RQ2_ANALYSIS_RESULTS_PATH,
    RQ2_DISAGREEMENT_PATH,
    RQ2_TP_ALIGNMENT_PATH,
    full_qual_df,
    mo,
    os,
    pd,
    sampled_qual_df,
):
    _qual_dir   = os.path.join(OUTPUT_DIR, 'qualitative_analysis')
    _out_path   = os.path.join(_qual_dir, 'rq2_qualitative_sampled_enriched.jsonl')
    _full_out_path = os.path.join(_qual_dir, 'rq2_qualitative_full_enriched.jsonl')
    _JOIN_COLS  = ['ID', 'Metric', 'Eval Model', 'Dataset']
    _LLM_COLS   = ['model_response_parsed', 'prompt_type', 'parse_error', 'scoring_model', 'prompt_version', 'failure_mode']

    if not os.path.exists(RQ2_ANALYSIS_RESULTS_PATH):
        _display = mo.vstack([
            mo.md('## RQ2 Sampled JSONL — LLM Enrichment'),
            mo.callout(
                mo.md(f'Analysis results not found at `{RQ2_ANALYSIS_RESULTS_PATH}` — skipping enrichment.'),
                kind='warn',
            ),
        ])
    else:
        _analysis_df = pd.read_json(RQ2_ANALYSIS_RESULTS_PATH, lines=True)

        # Normalise pre-reframe metric names → current names before joining
        _metric_name_map = {
            'Analytical-Rubric':             'Omission-Harm Rubric (Zero-shot)',
            'Analytical-Rubric (FS)':        'Omission-Harm Rubric',
            'Analytical-Rubric (FSS)':       'Omission-Harm Rubric',
            'Analytical-Rubric (Prometheus)': 'Omission-Harm Rubric (Prometheus)',
            'Dynamic-Checklist':             'Criteria-Checklist',
        }
        _analysis_df['Metric'] = _analysis_df['Metric'].replace(_metric_name_map)

        # TP reasoning alignment comes from the final v5 classifier (TODO §4a). For the
        # MedExpert TP rows it covers, the v5 output replaces the old classifier output, with
        # the code-derived label stored as `omissions_aligned` so every reader below picks it
        # up. The zero-shot Omission-Harm TP rows were not rerun and are excluded from all
        # alignment results; their old label is cleared so the two classifiers never mix.
        _v5_df = pd.read_json(RQ2_TP_ALIGNMENT_PATH, lines=True)
        _v5_df['model_response_parsed'] = [
            {**(_p if isinstance(_p, dict) else {}), 'omissions_aligned': _lbl}
            for _p, _lbl in zip(_v5_df['model_response_parsed'], _v5_df['omissions_aligned_derived'])
        ]
        _v5_df['scoring_model'] = _v5_df['served_model']
        _v5_cols = ['model_response_parsed', 'parse_error', 'scoring_model', 'prompt_version']
        _analysis_df = _analysis_df.merge(
            _v5_df[_JOIN_COLS + _v5_cols], on=_JOIN_COLS, how='left', suffixes=('', '_v5'),
            validate='one_to_one',
        )
        _is_v5 = _analysis_df['prompt_version'].notna()
        for _c in ['model_response_parsed', 'parse_error', 'scoring_model']:
            _analysis_df[_c] = _analysis_df[f'{_c}_v5'].where(_is_v5, _analysis_df[_c])
        _analysis_df = _analysis_df.drop(columns=[f'{_c}_v5' for _c in ['model_response_parsed', 'parse_error', 'scoring_model']])
        _analysis_df['prompt_version'] = _analysis_df['prompt_version'].fillna('v0')
        assert _is_v5.sum() == len(_v5_df), f'{_is_v5.sum()} of {len(_v5_df)} v5 rows joined'

        _medexpert_tp = (
            (_analysis_df['Dataset'] == 'MedExpert')
            & _analysis_df['Human Judgment'].str.startswith('Incomplete')
            & _analysis_df['Model Judgment'].str.startswith('Incomplete')
        )
        _stale_tp = _medexpert_tp & ~_is_v5
        assert (_analysis_df.loc[_stale_tp, 'Metric'] == 'Omission-Harm Rubric (Zero-shot)').all(), \
            _analysis_df.loc[_stale_tp, 'Metric'].value_counts()
        _analysis_df['model_response_parsed'] = [
            {**_p, 'omissions_aligned': None} if _stale and isinstance(_p, dict) else _p
            for _p, _stale in zip(_analysis_df['model_response_parsed'], _stale_tp)
        ]
        _n_v5, _n_stale = int(_is_v5.sum()), int(_stale_tp.sum())

        # FP/FN failure modes come from the final d1 classifier (TODO §4c), as a separate
        # `failure_mode` column. It covers every MedExpert FP/FN row under both rubrics.
        _d1_df = pd.read_json(RQ2_DISAGREEMENT_PATH, lines=True)
        assert _d1_df['disagreement_label_derived'].notna().all(), 'd1 row without a label'
        _analysis_df = _analysis_df.merge(
            _d1_df[_JOIN_COLS + ['disagreement_label_derived']].rename(columns={'disagreement_label_derived': 'failure_mode'}),
            on=_JOIN_COLS, how='left', validate='one_to_one',
        )
        _n_d1 = int(_analysis_df['failure_mode'].notna().sum())
        assert _n_d1 == len(_d1_df), f'{_n_d1} of {len(_d1_df)} d1 rows joined'

        # Keep only the join key + LLM fields that are present in the file
        _keep = _JOIN_COLS + [c for c in _LLM_COLS if c in _analysis_df.columns]
        _analysis_df = _analysis_df[_keep].drop_duplicates(subset=_JOIN_COLS)

        _enriched_df = sampled_qual_df.merge(_analysis_df, on=_JOIN_COLS, how='left')

        _n_total = len(_enriched_df)
        _n_matched = _enriched_df['model_response_parsed'].notna().sum() if 'model_response_parsed' in _enriched_df.columns else 0

        _enriched_df.to_json(_out_path, orient='records', lines=True, force_ascii=False)

        # Full enriched export (all qualitative rows, not just sample)
        _full_enriched_df = full_qual_df.rename(columns={'_category': 'analysis_category'}).merge(
            _analysis_df, on=_JOIN_COLS, how='left'
        )
        _full_enriched_df.to_json(_full_out_path, orient='records', lines=True, force_ascii=False)
        _n_full_matched = _full_enriched_df['model_response_parsed'].notna().sum() if 'model_response_parsed' in _full_enriched_df.columns else 0

        # Demo sample: up to 10 rows with LLM analysis, stratified across
        # analysis_category. The second sample must be sized against the stratified
        # frame, not the pool it came from — with two rows per category the stratified
        # frame is much the smaller of the two.
        _matched_df = _enriched_df[_enriched_df['model_response_parsed'].notna()]
        _demo_strata = (
            _matched_df
            .groupby('analysis_category', group_keys=False)
            .apply(lambda g: g.sample(n=min(2, len(g)), random_state=42))
        )
        _demo_df = (
            _demo_strata
            .sample(n=min(10, len(_demo_strata)), random_state=42)
            .reset_index(drop=True)
        )
        _demo_path = os.path.join(_qual_dir, 'rq2_qualitative_demo.jsonl')
        _demo_df.to_json(_demo_path, orient='records', lines=True, force_ascii=False)

        # Unmatched breakdown by Dataset / Metric / Eval Model
        _unmatched = _enriched_df[_enriched_df['model_response_parsed'].isna()] \
            if 'model_response_parsed' in _enriched_df.columns \
            else _enriched_df
        _breakdown = (
            _unmatched
            .groupby(['Dataset', 'Metric', 'Eval Model'], dropna=False)
            .size()
            .reset_index(name='unmatched')
            .sort_values('unmatched', ascending=False)
        )
        _breakdown_md = _breakdown.to_markdown(index=False) if not _breakdown.empty else '_none_'

        _display = mo.vstack([
            mo.md('## RQ2 Sampled JSONL — LLM Enrichment'),
            mo.md(f'- **{_n_matched} / {_n_total}** sampled rows matched to LLM analysis results'),
            mo.md(f'- **{_n_full_matched:,} / {len(_full_enriched_df):,}** full rows matched'),
            mo.md(f'- LLM fields added: {", ".join(f"`{c}`" for c in _LLM_COLS)}'),
            mo.md(f'- TP alignment: **{_n_v5:,}** MedExpert TP rows use the v5 classifier label; '
                  f'{_n_stale:,} zero-shot Omission-Harm TP rows (not rerun) have their old label cleared'),
            mo.md(f'- FP/FN failure modes: **{_n_d1:,}** MedExpert FP/FN rows carry the d1 label in `failure_mode`'),
            mo.md(f'Sampled enriched JSONL saved to `{_out_path}`'),
            mo.md(f'Full enriched JSONL saved to `{_full_out_path}`'),
            mo.md(f'Demo sample ({len(_demo_df)} rows) saved to `{_demo_path}`'),
            mo.md('### Unmatched rows (sampled) by Dataset / Metric / Eval Model'),
            mo.md(_breakdown_md),
        ])
    full_enriched_df = _full_enriched_df if os.path.exists(RQ2_ANALYSIS_RESULTS_PATH) else None
    _display
    return (full_enriched_df,)


@app.cell
def _():
    # d1 failure-mode labels (`failure_mode`), in plotting order, with the display names
    # used in tab:disagreement_examples. FP: first matching rule wins, top to bottom.
    FP_FAILURE_MODES = [
        'misreads_question', 'already_addressed', 'same_concern_weighted_differently',
        'topic:seek_care', 'topic:patient_information', 'topic:clinical_detail', 'topic:other',
    ]
    FN_FAILURE_MODES = ['names_as_missing', 'considers_covered', 'not_mentioned']
    FAILURE_MODE_NAMES = {
        'misreads_question': 'Misreads question',
        'already_addressed': 'Already addressed',
        'same_concern_weighted_differently': 'Same concern, weighted differently',
        'topic:seek_care': 'New concern: when to seek care',
        'topic:patient_information': 'New concern: patient information',
        'topic:clinical_detail': 'New concern: clinical detail',
        'topic:other': 'New concern: other',
        'names_as_missing': 'Names as missing',
        'considers_covered': 'Considers covered',
        'not_mentioned': 'Not mentioned',
    }
    FAILURE_MODE_COLORS = {
        'misreads_question': '#7f7f7f',
        'already_addressed': '#8c564b',
        'same_concern_weighted_differently': '#9467bd',
        'topic:seek_care': '#d62728',
        'topic:patient_information': '#ff7f0e',
        'topic:clinical_detail': '#bcbd22',
        'topic:other': '#e377c2',
        'names_as_missing': '#2ca02c',
        'considers_covered': '#17becf',
        'not_mentioned': '#1f77b4',
    }
    return FAILURE_MODE_COLORS, FAILURE_MODE_NAMES, FN_FAILURE_MODES, FP_FAILURE_MODES


@app.cell(hide_code=True)
def _(mo):
    n_validation_examples_widget = mo.ui.number(
        start=1, stop=50, step=1, value=5,
        label="**Validation sample: N per stratum (analysis_category × assigned_class)**",
    )
    mo.output.replace(n_validation_examples_widget)
    return (n_validation_examples_widget,)


@app.cell
def _(
    OUTPUT_DIR,
    full_enriched_df,
    json,
    mo,
    n_validation_examples_widget,
    os,
    pd,
    random,
    shutil,
    warnings,
):

    _n = n_validation_examples_widget.value
    _qual_dir = os.path.join(OUTPUT_DIR, 'qualitative_analysis')
    _annotated_dir = os.path.join(_qual_dir, 'annotated')
    _tools_dir = os.path.normpath(os.path.join(OUTPUT_DIR, '..', '..', '..', 'tools'))

    if full_enriched_df is None:
        mo.vstack([
            mo.md('## RQ2 Validation Sample'),
            mo.callout(mo.md('Full enriched dataframe not available — skipping.'), kind='warn'),
        ])
    else:
        # --- Load already-annotated IDs to exclude from new sample ---
        _ann_files = [
            'rq2_qualitative_validation_fp_annotated.jsonl',
            'rq2_qualitative_validation_fn_evened_annotated.jsonl',
            'rq2_qualitative_validation_disagrees_annotated.jsonl',
            'rq2_qualitative_validation_tp_agrees_evened_annotated.jsonl',
        ]
        _annotated_ids = set()
        for _af in _ann_files:
            _ap = os.path.join(_annotated_dir, _af)
            if os.path.exists(_ap):
                with open(_ap) as _f:
                    for _line in _f:
                        if _line.strip():
                            _r = json.loads(_line)
                            # Annotated = human_category filled (FP/FN)
                            #           or human_omissions_aligned filled (TP)
                            _is_annotated = (
                                _r.get('human_category', '').strip() or
                                str(_r.get('human_omissions_aligned') or '').strip()
                            )
                            if _r.get('ID') and _is_annotated:
                                _annotated_ids.add(_r['ID'])

        # --- Filter pool: exclude TN, ZS, already-annotated ---
        _df = full_enriched_df[
            full_enriched_df['model_response_parsed'].notna() &
            (full_enriched_df['Metric'] != 'Omission-Harm Rubric (Zero-shot)') &
            (full_enriched_df['analysis_category'] != 'TN') &
            (~full_enriched_df['ID'].isin(_annotated_ids))
        ].copy()

        def _extract_assigned_class(row):
            parsed = row.get('model_response_parsed') or {}
            if not isinstance(parsed, dict): parsed = {}
            if row.get('prompt_type') == 'same_verdict':
                return parsed.get('omissions_aligned', None)
            else:
                return parsed.get('category', None)

        _df['assigned_class'] = _df.apply(_extract_assigned_class, axis=1)

        # Filter cross-category mismatches (e.g. FP rows with FN sub-codes)
        _FP_CODES_VAL = {f'FP0{i}' for i in range(1, 6)}
        _FN_CODES_VAL = {f'FN0{i}' for i in range(1, 7)}
        _ALIGN_VAL    = {'Yes', 'Partially', 'No'}
        _EXPECTED_VAL = {
            'FP': _FP_CODES_VAL, 'FN': _FN_CODES_VAL, 'TP': _ALIGN_VAL,
        }
        _df = _df[_df.apply(
            lambda r: r.get('assigned_class') in _EXPECTED_VAL.get(r.get('analysis_category', ''), set()),
            axis=1
        )]

        # --- Sample TP/FP/FN: N per (analysis_category × assigned_class) ---
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            _sampled = (
                _df
                .groupby(['analysis_category', 'assigned_class'], group_keys=False)
                .apply(lambda g: g.sample(n=min(_n, len(g)), random_state=42))
                .reset_index(drop=True)
            )

        def _flatten_row(row):
            parsed = row.get('model_response_parsed') or {}
            return {
                'ID':                             row.get('ID'),
                'Dataset':                        row.get('Dataset'),
                'Metric':                         row.get('Metric'),
                'Eval Model':                     row.get('Eval Model'),
                'Pairing Type':                   row.get('Pairing Type'),
                'analysis_category':              row.get('analysis_category'),
                'prompt_type':                    row.get('prompt_type'),
                'assigned_class':                 row.get('assigned_class'),
                'Question':                       row.get('Question'),
                'Answer':                         row.get('Answer'),
                'Human Judgment':                 row.get('Human Judgment'),
                'Model Judgment':                 row.get('Model Judgment'),
                'Explanation of Human Judgment':  row.get('Explanation of Human Judgment'),
                'Explanation of Model Judgment':  row.get('Explanation of Model Judgment'),
                'omissions_aligned':              parsed.get('omissions_aligned'),
                'category':                       parsed.get('category'),
                'clinician_omissions':            parsed.get('clinician_omissions') or parsed.get('clinician_key_concern'),
                'model_omissions':                parsed.get('model_omissions') or parsed.get('model_engagement_with_concern'),
                'llm_explanation':                parsed.get('explanation'),
                'scoring_model':                  row.get('scoring_model'),
                'human_label_correct':            '',
                'human_notes':                    '',
            }

        # --- Split into 4 annotator files (round-robin for category mix) ---
        _all_rows = [_flatten_row(r) for _, r in _sampled.iterrows()]
        _rng = random.Random(42)
        _rng.shuffle(_all_rows)
        _annotator_paths = []
        for _i in range(4):
            _chunk = _all_rows[_i::4]
            _fname = f'rq2_qualitative_validation_annotator{_i + 1}.jsonl'
            _fpath = os.path.join(_qual_dir, _fname)
            pd.DataFrame(_chunk).to_json(_fpath, orient='records', lines=True, force_ascii=False)
            shutil.copy2(_fpath, os.path.join(_tools_dir, _fname))
            _annotator_paths.append((_fname, len(_chunk)))

        # --- Strata summary ---
        _strata = (
            _df.groupby(['analysis_category', 'assigned_class'])
            .size().reset_index(name='pool')
            .merge(
                _sampled.groupby(['analysis_category', 'assigned_class'])
                .size().reset_index(name='sampled'),
                on=['analysis_category', 'assigned_class'],
            )
            .sort_values(['analysis_category', 'assigned_class'])
        )

        _ann_summary = '\n'.join(
            f'- `{fname}`: **{n} rows**'
            for fname, n in _annotator_paths
        )
        mo.vstack([
            mo.md('## RQ2 Validation Sample'),
            mo.md(
                f'**{len(_all_rows)} new rows** ({_n}/stratum, excluding {len(_annotated_ids)} already-annotated IDs), '
                f'split across 4 annotator files:'
            ),
            mo.md(_ann_summary),
            mo.md(f'Files also copied to `{_tools_dir}/`'),
            mo.ui.table(_strata, page_size=30),
        ])
    return


@app.cell
def _(OUTPUT_DIR, mo, os, pd):
    _qual_dir = os.path.join(OUTPUT_DIR, 'qualitative_analysis')
    _FP_CODES   = {f'FP0{i}' for i in range(1, 6)}
    _FN_CODES   = {f'FN0{i}' for i in range(1, 7)}
    _ALIGN      = {'Yes', 'Partially', 'No'}
    _EXPECTED   = {
        'FP':        _FP_CODES,
        'FN':        _FN_CODES,
        'TP':        _ALIGN,
    }

    _summary_rows = []
    _detail_rows  = []
    for _i in range(1, 5):
        _fname = f'rq2_qualitative_validation_annotator{_i}.jsonl'
        _fpath = os.path.join(_qual_dir, _fname)
        if not os.path.exists(_fpath):
            continue
        _df = pd.read_json(_fpath, lines=True)
        for _, _r in _df.iterrows():
            _cat   = str(_r.get('analysis_category') or '')
            _cls   = str(_r.get('assigned_class')    or '')
            _ok    = _cls in _EXPECTED.get(_cat, set())
            if not _ok:
                _detail_rows.append({
                    'File':               _fname,
                    'analysis_category':  _cat,
                    'assigned_class':     _cls,
                    'ID':                 _r.get('ID', ''),
                })
        _n_mismatch = sum(
            1 for _, _r in _df.iterrows()
            if str(_r.get('assigned_class') or '') not in _EXPECTED.get(str(_r.get('analysis_category') or ''), set())
        )
        _cat_counts = _df['analysis_category'].value_counts().to_dict() if 'analysis_category' in _df.columns else {}
        _summary_rows.append({
            'File':       _fname,
            'Total':      len(_df),
            'FP':         _cat_counts.get('FP', 0),
            'FN':         _cat_counts.get('FN', 0),
            'TP':         _cat_counts.get('TP', 0),
            'Mismatched': _n_mismatch,
        })

    _summary_df = pd.DataFrame(_summary_rows)
    _detail_df  = pd.DataFrame(_detail_rows) if _detail_rows else pd.DataFrame(
        columns=['File', 'analysis_category', 'assigned_class', 'ID']
    )

    mo.vstack([
        mo.md('### Annotator file summary'),
        mo.ui.table(_summary_df),
        mo.md('**Label mismatches:**'),
        mo.callout(mo.md('No mismatches found.'), kind='success') if _detail_df.empty
        else mo.ui.table(_detail_df, page_size=20),
    ])
    return


@app.cell
def _(full_enriched_df, mo):
    if full_enriched_df is None:
        mo.callout(mo.md('Full enriched dataframe not available — skipping.'), kind='warn')
    else:
        _FP_CODES = {f'FP0{i}' for i in range(1, 6)}
        _FN_CODES = {f'FN0{i}' for i in range(1, 7)}
        _ALIGN    = {'Yes', 'Partially', 'No'}
        _EXPECTED = {
            'FP':        _FP_CODES,
            'FN':        _FN_CODES,
            'TP':        _ALIGN,
            'TN':        _ALIGN,
        }

        # Only rows that have an LLM-assigned label and are not zero-shot
        _df = full_enriched_df[
            full_enriched_df['model_response_parsed'].notna() &
            (full_enriched_df['Metric'] != 'Omission-Harm Rubric (Zero-shot)')
        ].copy()

        def _extract_cls(row):
            p = row.get('model_response_parsed') or {}
            return p.get('omissions_aligned') if row.get('prompt_type') == 'same_verdict' else p.get('category')

        _df['assigned_class'] = _df.apply(_extract_cls, axis=1)

        _df['_mismatch'] = _df.apply(
            lambda r: str(r.get('assigned_class') or '') not in _EXPECTED.get(str(r.get('analysis_category') or ''), set()),
            axis=1
        )

        _total = len(_df)
        _n_mismatch = _df['_mismatch'].sum()

        # Summary by analysis_category
        _by_cat = (
            _df.groupby('analysis_category')
            .agg(total=('_mismatch', 'count'), mismatched=('_mismatch', 'sum'))
            .reset_index()
        )
        _by_cat['pct'] = (_by_cat['mismatched'] / _by_cat['total'] * 100).round(1).astype(str) + '%'
        _by_cat = _by_cat.rename(columns={'analysis_category': 'Category', 'total': 'Total', 'mismatched': 'Mismatched', 'pct': 'Mismatch %'})

        # Breakdown of mismatch types: analysis_category × assigned_class
        _mismatch_df = _df[_df['_mismatch']].copy()
        _breakdown = (
            _mismatch_df.groupby(['analysis_category', 'assigned_class'])
            .size().reset_index(name='Count')
            .sort_values(['analysis_category', 'Count'], ascending=[True, False])
            .rename(columns={'analysis_category': 'Category', 'assigned_class': 'Assigned Class'})
        )

        mo.vstack([
            mo.md('### Cross-category label mismatches — full dataset'),
            mo.md(f'**{_n_mismatch:,} / {_total:,}** labeled rows have a mismatched `assigned_class` '
                  f'({_n_mismatch / _total * 100:.1f}%).'),
            mo.ui.table(_by_cat),
            mo.md('**Mismatch breakdown by category × assigned class:**'),
            mo.callout(mo.md('No mismatches found.'), kind='success') if _breakdown.empty
            else mo.ui.table(_breakdown),
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## RQ2 Figures
    """)
    return


@app.cell
def _(OUTPUT_DIR, full_enriched_df, mo, os, pd, plt, save_figure, sns):
    if full_enriched_df is None:
        _out = mo.md('⚠️ No enriched data — skipping RQ2 figures.')
    else:

        # ── Omission-alignment heatmap broken down by LLM Judge backbone ──────
        # Incomplete-agree pairs only (both human and model judged Incomplete).
        _ALIGN_ORDER    = ['Yes', 'Partially', 'No']
        # OpenBioLLM-70B is reported separately in the appendix (app:openbiollm)
        # and excluded from the main-paper figures.
        _MAIN_BACKBONES = ['LLaMA 3.3-70B', 'GPT-5 Mini', 'Qwen3.6', 'Prometheus-2 7B']
        _BACKBONE_LABELS = {
            'LLaMA 3.3-70B':   'LLaMA',
            'GPT-5 Mini':      'GPT-5',
            'Qwen3.6':         'Qwen3.6',
            'Prometheus-2 7B': 'Prom-2',
        }

        # Both completeness definitions on MedExpert, against the same clinician
        # annotations. Verdict-level performance is indistinguishable (RQ1); these
        # panels show that reasoning alignment is not.
        _PAIRINGS = [
            ('TP', 'Omission-Harm Rubric', 'MedExpert', 'Omission-Harm Rubric'),
            ('TP', 'Criteria-Checklist',   'MedExpert', 'Criteria-Checklist'),
        ]

        # Panels stacked vertically so the figure fits one column. A dedicated
        # colorbar axis spanning both rows keeps the heatmaps the same size;
        # letting seaborn steal space from the last panel makes it narrower.
        _fig = plt.figure(figsize=(3.2, 4.4))
        _gs = _fig.add_gridspec(len(_PAIRINGS), 2, width_ratios=[1, 0.05], wspace=0.08, hspace=0.45)
        _axes = [_fig.add_subplot(_gs[_j, 0]) for _j in range(len(_PAIRINGS))]
        _cax = _fig.add_subplot(_gs[:, 1])

        for _i, (_ax, (_cat, _metric, _dataset, _title)) in enumerate(zip(_axes, _PAIRINGS)):
            _df_pair = full_enriched_df[
                (full_enriched_df['analysis_category'] == _cat) &
                (full_enriched_df['Metric'] == _metric) &
                (full_enriched_df['Dataset'] == _dataset) &
                full_enriched_df['Human Judgment'].str.startswith('Incomplete') &
                full_enriched_df['Model Judgment'].str.startswith('Incomplete') &
                full_enriched_df['Eval Model'].isin(_MAIN_BACKBONES)
            ].copy()
            _df_pair['aligned'] = _df_pair['model_response_parsed'].apply(
                lambda x: x.get('omissions_aligned') if isinstance(x, dict) else None
            )

            _rows = {}
            _counts = {}
            for _backbone in _MAIN_BACKBONES:
                _bsub = _df_pair[_df_pair['Eval Model'] == _backbone]['aligned']
                _total = len(_bsub)
                _n_by_align = {_a: int((_bsub == _a).sum()) for _a in _ALIGN_ORDER}
                # A backbone with no run under this rubric (Prometheus-2 7B has no
                # Criteria-Checklist run) gets NaN, so its row is left blank and
                # labelled "no run" below instead of reading as 0%.
                _rows[_BACKBONE_LABELS[_backbone]] = {
                    _a: _n_by_align[_a] / _total * 100 if _total else float('nan')
                    for _a in _ALIGN_ORDER
                }
                _counts[_BACKBONE_LABELS[_backbone]] = _n_by_align
            _hm = pd.DataFrame(_rows, index=_ALIGN_ORDER).T
            # Cells carry both the row-normalised percentage and the raw count,
            # so the reader can see how thin some backbone/alignment cells are.
            _hm_labels = pd.DataFrame(_counts, index=_ALIGN_ORDER).T.astype(int)
            _annot = pd.DataFrame(
                [[
                    '' if pd.isna(_hm.loc[_r, _a]) else f'{_hm.loc[_r, _a]:.0f}%\n({_hm_labels.loc[_r, _a]})'
                    for _a in _ALIGN_ORDER
                ] for _r in _hm.index],
                index=_hm.index, columns=_ALIGN_ORDER,
            )

            sns.heatmap(
                _hm,
                ax=_ax,
                annot=_annot,
                fmt='',
                cmap='Blues',
                vmin=0, vmax=100,
                cbar=(_i == len(_PAIRINGS) - 1),
                cbar_ax=_cax if _i == len(_PAIRINGS) - 1 else None,
                cbar_kws={'label': '%'} if _i == len(_PAIRINGS) - 1 else {},
                linewidths=0.4,
                annot_kws={'size': 10},
            )
            for _r_idx, _r in enumerate(_hm.index):
                if _hm.loc[_r].isna().all():
                    _ax.text(
                        len(_ALIGN_ORDER) / 2, _r_idx + 0.5, 'no run',
                        ha='center', va='center', fontsize=9, style='italic', color='0.4',
                    )
            _ax.set_xticklabels(_ALIGN_ORDER, fontsize=9, rotation=0)
            _ax.set_yticklabels(_hm.index.tolist(), rotation=0, fontsize=9)
            _ax.set_xlabel('Reasoning Alignment' if _i == len(_PAIRINGS) - 1 else '', fontsize=9)
            _ax.set_ylabel('')
            _ax.set_title(_title, fontsize=11)

        _cax.tick_params(labelsize=8)
        _cax.set_ylabel('%', fontsize=9)
        _fig.tight_layout()
        save_figure(_fig, os.path.join(OUTPUT_DIR, 'rq2_omission_coverage'))
        _out = mo.vstack([_fig])
    _out
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verdict Performance vs. Reasoning Alignment, by Completeness Definition

    The paper's central comparison. Both completeness definitions are applied to the same
    clinician-annotated MedExpert responses, so verdict-level performance and reasoning alignment can be
    read side by side. Alignment is restricted to pairs where clinician *and* judge called the response
    Incomplete, since only those have omissions on both sides to compare.

Alignment is restricted with the two verdict columns directly rather than through
    `analysis_category`.
    """)
    return


@app.cell
def _(
    ID_TO_QID,
    OUTPUT_DIR,
    SETTINGS,
    full_enriched_df,
    make_cluster_sampler,
    mo,
    np,
    os,
    pd,
):
    if full_enriched_df is None:
        alignment_by_rubric_df = pd.DataFrame()
        ALIGNMENT_ROW_POOLS = []
        _align_out = mo.md('⚠️ No enriched data — skipping alignment-by-rubric table.')
    else:
        _ALIGN_ORDER = ['Yes', 'Partially', 'No']
        _RUBRICS = ['Omission-Harm Rubric', 'Criteria-Checklist']
        _judge_order = [s['model'] for s in SETTINGS if s['model'] != 'Prometheus-2 BGb 8x7B']
        _N_RESAMPLES = 10000

        _aligned = full_enriched_df[
            (full_enriched_df['Dataset'] == 'MedExpert') &
            full_enriched_df['Metric'].isin(_RUBRICS) &
            full_enriched_df['Human Judgment'].str.startswith('Incomplete') &
            full_enriched_df['Model Judgment'].str.startswith('Incomplete')
        ].copy()
        _aligned['aligned'] = _aligned['model_response_parsed'].apply(
            lambda x: x.get('omissions_aligned') if isinstance(x, dict) else None
        )
        _aligned['qid'] = _aligned['ID'].map(ID_TO_QID)
        assert _aligned['qid'].notna().all(), 'TP row without a question cluster'

        # OpenBioLLM is reported but excluded from the main-text prose, so the
        # pooled row is given both with and without it.
        _EXCLUDED_FROM_PROSE = 'OpenBioLLM'
        _SHARED_JUDGES = ['LLaMA 3.3-70B', 'GPT-5 Mini', 'Qwen3.6']

        # (rubric, row key, member backbones) for every table row. A pooled row whose
        # members (among the backbones run with this rubric) equal the shared-backbone
        # pool is dropped, since it would repeat the Shared backbones row (for the
        # Criteria-Checklist, "All excl. OpenBioLLM" is exactly the shared three).
        ALIGNMENT_ROW_POOLS = []
        for _rubric in _RUBRICS:
            _run = sorted(_aligned.loc[_aligned['Metric'] == _rubric, 'Eval Model'].unique())
            _shared_set = set(_SHARED_JUDGES) & set(_run)
            for _judge in _judge_order:
                if _judge in _run:
                    ALIGNMENT_ROW_POOLS.append((_rubric, _judge, [_judge]))
            _pooled = [
                ('All', list(_run)),
                ('All-excl-OpenBioLLM', [j for j in _run if j != _EXCLUDED_FROM_PROSE]),
            ]
            for _key, _members in _pooled:
                if set(_members) != _shared_set:
                    ALIGNMENT_ROW_POOLS.append((_rubric, _key, _members))
            # The three backbones run with both definitions, so the pooled rates of the
            # two definitions describe the same Judges.
            ALIGNMENT_ROW_POOLS.append((_rubric, 'Shared', _SHARED_JUDGES))

        # Rates with 95% question-clustered bootstrap CIs: responses to the same question
        # (across generators and backbones) are resampled together, as for Table 1.
        _rows_align = []
        for _ri, (_rubric, _key, _members) in enumerate(ALIGNMENT_ROW_POOLS):
            _a = _aligned[(_aligned['Metric'] == _rubric) & _aligned['Eval Model'].isin(_members)]
            _n = len(_a)
            _row = {'metric': _rubric, 'eval_model': _key, 'n_pairs': _n}
            if _n:
                _lab = _a['aligned'].values
                _draw = make_cluster_sampler(_a['qid'].values)
                _rng = np.random.default_rng(20261114 + _ri)
                _boot = np.array([
                    [(_lab[_idx] == _l).mean() for _l in _ALIGN_ORDER]
                    for _idx in (_draw(_rng) for _ in range(_N_RESAMPLES))
                ])
                _lo, _hi = np.percentile(_boot, [2.5, 97.5], axis=0)
                for _j, _l in enumerate(_ALIGN_ORDER):
                    _row[_l] = (_lab == _l).mean()
                    _row[f'{_l}_lo'] = _lo[_j]
                    _row[f'{_l}_hi'] = _hi[_j]
            _rows_align.append(_row)

        alignment_by_rubric_df = pd.DataFrame(_rows_align)
        _align_out = mo.ui.table(
            alignment_by_rubric_df.round(3), selection=None, page_size=len(alignment_by_rubric_df)
        )

        _path = os.path.join(OUTPUT_DIR, 'alignment_by_rubric.tex')
        _fmt_ci = lambda r, l: (
            '--' if pd.isna(r.get(l)) else
            f"{r[l] * 100:.1f} {{\\scriptsize [{r[l + '_lo'] * 100:.1f}, {r[l + '_hi'] * 100:.1f}]}}"
        )
        _JUDGE_TEX = {
            'All': r'\textit{All}',
            'All-excl-OpenBioLLM': r'\textit{All excl. OpenBioLLM}',
            'Shared': r'\textit{Shared backbones}',
            _EXCLUDED_FROM_PROSE: _EXCLUDED_FROM_PROSE + r'$^*$',
        }
        _rows = [
            r'\begin{table}[t]',
            r'\centering',
            r'\small',
            r'\begin{tabular}{llrccc}',
            r'    \toprule',
            r'    \textbf{Definition} & \textbf{Judge} & $\boldsymbol{n}$'
            r' & \textbf{Yes} & \textbf{Partially} & \textbf{No} \\',
            r'    \midrule',
        ]
        for _ri, _rubric in enumerate(_RUBRICS):
            _sub = alignment_by_rubric_df[alignment_by_rubric_df['metric'] == _rubric]
            if _sub.empty:
                continue
            _n_rows = len(_sub)
            _first_pooled = True
            for _i, (_, _r) in enumerate(_sub.iterrows()):
                _is_pooled = _r['eval_model'] in ('All', 'All-excl-OpenBioLLM', 'Shared')
                if _is_pooled and _first_pooled:
                    # Partial rule between the individual backbones and the pooled rows.
                    _rows.append(r'    \cmidrule(l){2-6}')
                    _first_pooled = False
                _label = r'\multirow{' + str(_n_rows) + r'}{*}{' + _rubric + r'}' if _i == 0 else ''
                _judge = _JUDGE_TEX.get(_r['eval_model'], _r['eval_model'])
                _rows.append(
                    f"    {_label} & {_judge} & {int(_r['n_pairs'])} & {_fmt_ci(_r, 'Yes')}"
                    f" & {_fmt_ci(_r, 'Partially')} & {_fmt_ci(_r, 'No')} \\\\"
                )
            if _ri < len(_RUBRICS) - 1:
                _rows.append(r'    \midrule')
        _rows += [
            r'    \bottomrule',
            r'\end{tabular}',
            r'\caption{Reasoning alignment of the True Positive pairs (clinician and Judge both rated the '
            r'response Incomplete) on MedExpert, as \% of the $n$ pairs: the Judge cited every clinician '
            r'concern (\textit{Yes}), at least one at least in part (\textit{Partially}), or none '
            r'(\textit{No}). Labels are from the GPT-5.4~Mini classifier. Brackets: 95\% question-clustered '
            r'bootstrap confidence intervals (' + f'{_N_RESAMPLES:,}'.replace(',', '{,}') + r' resamples '
            r'of the questions). \textit{All} pools every backbone run with the definition; '
            r'\textit{Shared backbones} pools LLaMA~3.3-70B, GPT-5~Mini and Qwen~3.6, the three backbones '
            r'run with both definitions. '
            r'$^*$OpenBioLLM-70B is reported for completeness but excluded '
            r'from the main-text analysis; see \Cref{app:openbiollm}.'
            r'}\label{tab:alignment_by_rubric}',
            r'\end{table}',
        ]
        with open(_path, 'w') as _f:
            _f.write('\n'.join(_rows) + '\n')
        print(f'Saved {_path}')
    _align_out
    return ALIGNMENT_ROW_POOLS, alignment_by_rubric_df




@app.cell
def _(
    FAILURE_MODE_COLORS,
    FAILURE_MODE_NAMES,
    FN_FAILURE_MODES,
    FP_FAILURE_MODES,
    OUTPUT_DIR,
    full_enriched_df,
    mo,
    os,
    plt,
    save_figure,
):
    if full_enriched_df is None:
        _out2 = mo.md('⚠️ No enriched data — skipping RQ2 failure mode figure.')
    else:
        # d1 failure modes, Omission-Harm Rubric only, all five backbones.
        _FP_CODES, _FN_CODES = FP_FAILURE_MODES, FN_FAILURE_MODES
        _FP_COLORS = _FN_COLORS = FAILURE_MODE_COLORS
        _CODE_NAMES = FAILURE_MODE_NAMES
        _BACKBONE_ORDER = ['LLaMA 3.3-70B', 'OpenBioLLM', 'GPT-5 Mini', 'Qwen3.6', 'Prometheus-2 7B']
        _BACKBONE_LABELS = {
            'LLaMA 3.3-70B': 'LLaMA 3.3-70B',
            'OpenBioLLM': 'OpenBioLLM-70B',
            'GPT-5 Mini': 'GPT-5 Mini',
            'Qwen3.6': 'Qwen 3.6',
            'Prometheus-2 7B': 'Prometheus-2 7B',
        }

        _df_disag = full_enriched_df[
            full_enriched_df['analysis_category'].isin(['FP', 'FN']) &
            (full_enriched_df['Metric'] == 'Omission-Harm Rubric')
        ].copy()
        _df_disag['code'] = _df_disag['failure_mode']
        _backbones_present = [b for b in _BACKBONE_ORDER if b in _df_disag['Eval Model'].unique()]

        _fig, _axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

        for _ax, (_verdict, _codes, _colors) in zip(
            _axes,
            [('FP', _FP_CODES, _FP_COLORS), ('FN', _FN_CODES, _FN_COLORS)],
        ):
            _sub = _df_disag[_df_disag['analysis_category'] == _verdict]

            for _bi, _backbone in enumerate(_backbones_present):
                _bsub = _sub[_sub['Eval Model'] == _backbone]['code']
                _total = len(_bsub)
                _left = 0.0
                for _code in _codes:
                    _n = _bsub.eq(_code).sum()
                    _pct = _n / _total * 100 if _total else 0
                    _ax.barh(
                        _bi, _pct, left=_left, height=0.6,
                        color=_colors[_code],
                        label=_CODE_NAMES[_code] if _bi == 0 else '',
                    )
                    if _pct >= 5:
                        _ax.text(
                            _left + _pct / 2, _bi,
                            f'{_pct:.0f}% ({_n})',
                            ha='center', va='center', fontsize=7,
                            color='white', fontweight='bold',
                        )
                    _left += _pct

            _ax.set_yticks(range(len(_backbones_present)))
            _ax.set_yticklabels([_BACKBONE_LABELS.get(b, b) for b in _backbones_present], fontsize=9)
            _ax.invert_yaxis()
            _ax.set_xlim(0, 100)
            _ax.set_title(
                'Over-penalization (FP)' if _verdict == 'FP' else 'Missed Omission (FN)',
                fontsize=10, loc='left',
            )
            _ax.legend(
                fontsize=8, ncol=2,
                loc='upper center', bbox_to_anchor=(0.5, -0.18),
                frameon=False,
            )
            _ax.spines['top'].set_visible(False)
            _ax.spines['right'].set_visible(False)

        _axes[-1].set_xlabel('Percentage (%)', fontsize=9)
        _fig.subplots_adjust(hspace=0.55)
        save_figure(_fig, os.path.join(OUTPUT_DIR, 'rq2_fp_fn_failure_modes'))
        _out2 = mo.vstack([_fig])
    _out2
    return



@app.cell
def _(
    FAILURE_MODE_COLORS,
    FAILURE_MODE_NAMES,
    FN_FAILURE_MODES,
    FP_FAILURE_MODES,
    OUTPUT_DIR,
    full_enriched_df,
    mo,
    os,
    pd,
    plt,
    save_figure,
):
    if full_enriched_df is None:
        _out_comb = mo.md('⚠️ No enriched data — skipping combined failure mode figure.')
        rq2_failure_mode_rates_df = None
    else:
        # d1 failure modes (`failure_mode`). Labels never assigned in the run (topic:other)
        # are dropped so they take no legend entry.
        _assigned = set(full_enriched_df['failure_mode'].dropna())
        _FP_CODES = [c for c in FP_FAILURE_MODES if c in _assigned]
        _FN_CODES = [c for c in FN_FAILURE_MODES if c in _assigned]
        _FP_COLORS = _FN_COLORS = FAILURE_MODE_COLORS
        _CODE_NAMES = {**FAILURE_MODE_NAMES, 'OTHER': 'Other / unclassified'}
        _OTHER_COLOR = '#bdbdbd'
        _unclassified_rows = []
        # OpenBioLLM-70B is reported separately in the appendix (app:openbiollm)
        # and excluded from the main-paper figures.
        _BACKBONE_ORDER = ['LLaMA 3.3-70B', 'GPT-5 Mini', 'Qwen3.6', 'Prometheus-2 7B']
        _BACKBONE_LABELS = {
            'LLaMA 3.3-70B': 'LLaMA 3.3-70B',
            'GPT-5 Mini': 'GPT-5 Mini',
            'Qwen3.6': 'Qwen 3.6',
            'Prometheus-2 7B': 'Prometheus-2 7B',
        }

        # ── Pairing 1: OR × MedExpert (in-dataset FP/FN) ──────────────────────
        _df_orme = full_enriched_df[
            full_enriched_df['analysis_category'].isin(['FP', 'FN']) &
            (full_enriched_df['Metric'] == 'Omission-Harm Rubric')
        ].copy()
        _df_orme['code'] = _df_orme['failure_mode']
        _df_orme['direction'] = _df_orme['analysis_category']  # already FP / FN

        # ── Pairing 2: CC × MedExpert (in-dataset FP/FN) ──────────────────────
        _df_ccme = full_enriched_df[
            full_enriched_df['analysis_category'].isin(['FP', 'FN']) &
            (full_enriched_df['Metric'] == 'Criteria-Checklist')
        ].copy()
        _df_ccme['code'] = _df_ccme['failure_mode']
        _df_ccme['direction'] = _df_ccme['analysis_category']  # already FP / FN

        # Rates for the prose: per (definition, direction, Judge), plus pooled rows. The
        # pools are reported with and without Prometheus-2 7B, which copies the
        # Omission-Harm few-shot explanation (TODO §4c) and inflates two FP labels.
        _rate_rows = []
        for _metric, _pdf in [('Omission-Harm Rubric', _df_orme), ('Criteria-Checklist', _df_ccme)]:
            for _verdict, _codes in [('FP', FP_FAILURE_MODES), ('FN', FN_FAILURE_MODES)]:
                _d = _pdf[(_pdf['direction'] == _verdict) & _pdf['Eval Model'].isin(_BACKBONE_ORDER)]
                _groups = [(b, _d[_d['Eval Model'] == b]) for b in _BACKBONE_ORDER]
                _groups += [
                    ('All four', _d),
                    ('All excl. Prometheus-2 7B', _d[_d['Eval Model'] != 'Prometheus-2 7B']),
                    ('Shared backbones', _d[_d['Eval Model'].isin(['LLaMA 3.3-70B', 'GPT-5 Mini', 'Qwen3.6'])]),
                ]
                for _judge, _g in _groups:
                    if len(_g) == 0:
                        continue
                    _vc = _g['code'].value_counts()
                    for _c in _codes:
                        _rate_rows.append({
                            'Metric': _metric, 'direction': _verdict, 'Judge': _judge, 'label': _c,
                            'n': int(_vc.get(_c, 0)), 'total': len(_g),
                            'pct': round(100 * _vc.get(_c, 0) / len(_g), 1),
                        })
        rq2_failure_mode_rates_df = pd.DataFrame(_rate_rows)
        rq2_failure_mode_rates_df.to_csv(os.path.join(OUTPUT_DIR, 'rq2_failure_mode_rates.csv'), index=False)

        # Both panels are MedExpert, so the dataset is left to the caption; spelling
        # it out in both titles overruns the 5.5in panel width.
        _PAIRINGS = [
            ('Omission-Harm Rubric', _df_orme),
            ('Criteria-Checklist',   _df_ccme),
        ]

        _fig_comb, _axes_comb = plt.subplots(
            2, len(_PAIRINGS), figsize=(5.5, 4.5), sharex=True
        )

        for _col, (_pairing_label, _pair_df) in enumerate(_PAIRINGS):
            # Every backbone keeps a row in every panel, even when a pairing has no
            # coded cases for it (e.g. Prometheus-2 under Criteria-Checklist); those
            # rows are drawn as a solid grey bar so the row ordering stays comparable
            # across panels.
            _backbones_present = list(_BACKBONE_ORDER)

            for _row, (_verdict, _codes, _colors) in enumerate([
                ('FP', _FP_CODES, _FP_COLORS),
                ('FN', _FN_CODES, _FN_COLORS),
            ]):
                _ax = _axes_comb[_row, _col]
                _sub = _pair_df[_pair_df['direction'] == _verdict]

                for _bi, _backbone in enumerate(_backbones_present):
                    _bsub = _sub[_sub['Eval Model'] == _backbone]['code']
                    _total = len(_bsub)
                    if _total == 0:
                        # Hollow, not grey: a solid grey here is indistinguishable
                        # from the 'Other / unclassified' segment below.
                        _ax.barh(
                            _bi, 100, height=0.6, facecolor='white',
                            edgecolor='#9e9e9e', linewidth=0.7, linestyle='--',
                        )
                        _ax.text(
                            50, _bi, 'no cases',
                            ha='center', va='center', fontsize=7,
                            color='#757575', style='italic',
                        )
                        continue
                    _left = 0.0
                    for _code in _codes:
                        _n = _bsub.eq(_code).sum()
                        _pct = _n / _total * 100 if _total else 0
                        _ax.barh(
                            _bi, _pct, left=_left, height=0.6,
                            color=_colors[_code],
                        )
                        if _pct >= 8:
                            _ax.text(
                                _left + _pct / 2, _bi,
                                f'{_pct:.0f}%\n({_n})',
                                ha='center', va='center', fontsize=6,
                                color='white', fontweight='bold',
                            )
                        _left += _pct

                    # Any row without a d1 label for this direction is drawn as one
                    # grey segment so the bar totals 100%. The d1 run labels every
                    # MedExpert FP/FN row (asserted in the enrichment cell), so this
                    # should stay empty; the table below the figure lists any that land here.
                    _n_other = int(_total - _bsub.isin(_codes).sum())
                    _pct_other = _n_other / _total * 100 if _total else 0
                    if _n_other:
                        _ax.barh(
                            _bi, _pct_other, left=_left, height=0.6,
                            color=_OTHER_COLOR,
                        )
                        if _pct_other >= 8:
                            _ax.text(
                                _left + _pct_other / 2, _bi,
                                f'{_pct_other:.0f}%\n({_n_other})',
                                ha='center', va='center', fontsize=6,
                                color='#333333', fontweight='bold',
                            )
                        _unclassified_rows.append({
                            'Pairing': _pairing_label,
                            'Direction': _verdict,
                            'Eval Model': _backbone,
                            'n': _total,
                            'unclassified': _n_other,
                            'unclassified %': round(_pct_other, 1),
                            'codes': dict(
                                _bsub[~_bsub.isin(_codes)].fillna('<no category>').value_counts()
                            ),
                        })

                _ax.set_yticks(range(len(_backbones_present)))
                if _col == 0:
                    _ax.set_yticklabels(
                        [_BACKBONE_LABELS.get(b, b) for b in _backbones_present], fontsize=7,
                    )
                    _ax.set_ylabel(
                        # Codebook wording (tab:disagreement_codebook). The paper reserves
                        # "over-penalization" for the two Judge-error labels.
                        'Judge-only concerns\n(FP)' if _verdict == 'FP' else 'Clinician-only concerns\n(FN)',
                        fontsize=8,
                    )
                else:
                    _ax.set_yticklabels([])
                    _ax.set_ylabel('')
                _ax.invert_yaxis()
                _ax.set_xlim(0, 100)
                _ax.tick_params(axis='x', labelsize=7)
                _ax.spines['top'].set_visible(False)
                _ax.spines['right'].set_visible(False)

                if _row == 0:
                    _ax.set_title(_pairing_label, fontsize=9.5, fontweight='bold', pad=4)
                if _row == 1:
                    _ax.set_xlabel('Percentage (%)', fontsize=8)

        _handles_fp = [
            plt.Rectangle((0, 0), 1, 1, color=_FP_COLORS[c], label=_CODE_NAMES[c])
            for c in _FP_CODES
        ]
        _handles_fn = [
            plt.Rectangle((0, 0), 1, 1, color=_FN_COLORS[c], label=_CODE_NAMES[c])
            for c in _FN_CODES
        ]
        _handles_other = [
            plt.Rectangle((0, 0), 1, 1, color=_OTHER_COLOR, label=_CODE_NAMES['OTHER'])
        ] if _unclassified_rows else []
        # One titled legend per direction (FP left, FN right), with the codebook group names
        # used on the y-axis. "Other / unclassified" (empty unless a row lacks a d1 label)
        # spans both directions, so it goes in the shorter FN legend.
        _legend_kw = dict(
            fontsize=6.5, title_fontsize=7, alignment='left', frameon=False, ncol=1,
        )
        _fig_comb.legend(
            handles=_handles_fp, title='Judge-only concerns (FP)',
            loc='upper right', bbox_to_anchor=(0.54, 0.0), **_legend_kw,
        )
        _fig_comb.legend(
            handles=_handles_fn + _handles_other, title='Clinician-only concerns (FN)',
            loc='upper left', bbox_to_anchor=(0.58, 0.0), **_legend_kw,
        )
        _fig_comb.tight_layout()
        save_figure(_fig_comb, os.path.join(OUTPUT_DIR, 'rq2_rq3_failure_modes_combined'))

        _unclass_md = (
            pd.DataFrame(_unclassified_rows).to_markdown(index=False)
            if _unclassified_rows else '_every bar fully classified_'
        )
        _rates_wide = rq2_failure_mode_rates_df.pivot_table(
            index=['Metric', 'direction', 'label'], columns='Judge', values='pct', sort=False,
        )
        _out_comb = mo.vstack([
            _fig_comb,
            mo.md('### Unclassified segments (bars would otherwise fall short of 100%)'),
            mo.md(_unclass_md),
            mo.md('### Failure-mode rates (%) by Judge'),
            mo.ui.table(_rates_wide.reset_index(), selection=None),
        ])
    _out_comb
    return (rq2_failure_mode_rates_df,)


@app.cell
def _(full_enriched_df, mo):
    if full_enriched_df is None:
        _out4 = mo.md('⚠️ No enriched data — skipping RQ2 code counts table.')
    else:
        _df_codes = full_enriched_df.copy()
        _df_codes['code'] = _df_codes['model_response_parsed'].apply(
            lambda x: x.get('category') if isinstance(x, dict) else None
        )
        _counts = (
            _df_codes[_df_codes['code'].notna()]
            .groupby(['Dataset', 'Metric', 'analysis_category', 'code'])
            .size()
            .reset_index(name='count')
            .sort_values(['Dataset', 'Metric', 'analysis_category', 'code'])
        )
        _out4 = mo.vstack([
            mo.md('### LLM-Assigned Code Counts (Dataset × Rubric × Category × Code)'),
            mo.ui.dataframe(_counts),
        ])
    _out4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## RQ2 — Human Validation of Taxonomy Labels
    """)
    return


@app.cell
def _(MAIN_DATASETS, json):
    # The annotated validation files predate the retirement of the in-/cross-dataset
    # split, so their `analysis_category` still carries the Agrees/Disagrees vocabulary
    # for what were then cross-dataset pairings. Normalise on load: keep only the
    # datasets the paper reports, and recover TP/TN/FP/FN from the two verdicts so the
    # Criteria-Checklist rows sit in the same strata as the Omission-Harm rows.
    _CATEGORY_BY_VERDICT = {
        ('Incomplete', 'Incomplete'): 'TP',
        ('Complete',   'Complete'):   'TN',
        ('Complete',   'Incomplete'): 'FP',
        ('Incomplete', 'Complete'):   'FN',
    }

    def _verdict(row, key):
        # Stored as e.g. "Incomplete (0.25)" — the leading word is the verdict.
        return str(row.get(key) or '').split(' ')[0].strip()

    def normalize_validation_category(row):
        """TP/TN/FP/FN for a validation row, or None if the verdicts are unreadable."""
        pair = (_verdict(row, 'Human Judgment'), _verdict(row, 'Model Judgment'))
        return _CATEGORY_BY_VERDICT.get(pair)

    def load_validation_rows(path):
        """Annotated validation rows for MAIN_DATASETS, with categories normalised."""
        rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('Dataset') not in MAIN_DATASETS:
                    continue
                cat = normalize_validation_category(row)
                if cat is None:
                    continue
                row['analysis_category'] = cat
                rows.append(row)
        return rows

    return load_validation_rows, normalize_validation_category


@app.cell
def _(OUTPUT_DIR, load_validation_rows, mo, np, os, pd, simpledorff):
    _ANNOTATED_DIR = os.path.join(OUTPUT_DIR, 'qualitative_analysis', 'annotated')
    _FP_PATH = os.path.join(_ANNOTATED_DIR, 'rq2_qualitative_validation_fp_resampled_annotated.jsonl')
    _FN_PATH = os.path.join(_ANNOTATED_DIR, 'rq2_qualitative_validation_fn_resampled_annotated.jsonl')
    _DIS_PATH = os.path.join(_ANNOTATED_DIR, 'rq2_qualitative_validation_disagrees_resampled_annotated.jsonl')
    _TP_PATH = os.path.join(_ANNOTATED_DIR, 'rq2_qualitative_validation_tp_agrees_resampled_annotated.jsonl')

    def _same_type(a, b):
        return isinstance(a, str) and isinstance(b, str) and len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]

    def _compute_alpha(rows, assigned_col, human_col):
        records = []
        for i, r in enumerate(rows):
            item_id = r.get('ID') or r.get('id') or str(i)
            assigned = r.get(assigned_col)
            human = r.get(human_col) if str(r.get(human_col) or '').strip() else None
            records.append({'id': item_id, 'annotator': 'llm', 'label': assigned})
            records.append({'id': item_id, 'annotator': 'human', 'label': human})
        melted = pd.DataFrame(records)
        try:
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                melted, experiment_col='id', annotator_col='annotator', class_col='label'
            )
        except Exception as _e:
            print(f'Alpha error: {_e}')
            alpha = float('nan')
        return alpha

    def _load_fpfn(primary_path, extra_path, category):
        rows = load_validation_rows(primary_path) + load_validation_rows(extra_path)
        return [r for r in rows if r['analysis_category'] == category]

    def _stats_for_file(rows, category_filter=None):
        if category_filter:
            rows = [r for r in rows if r.get('analysis_category') == category_filter]
        annotated = [r for r in rows if r.get('human_category', '').strip()]
        n_total = len(rows)
        n_annotated = len(annotated)
        n_flagged = sum(1 for r in annotated if r.get('human_flagged_for_review') in (True, 'True', 'true', '1', 1))
        n_exact = sum(1 for r in annotated if r.get('assigned_class', '').strip() == r.get('human_category', '').strip())
        n_same_type = sum(1 for r in annotated if _same_type(r.get('assigned_class', ''), r.get('human_category', '')))
        alpha = _compute_alpha(rows, 'assigned_class', 'human_category')
        return {
            'n_total': n_total,
            'n_annotated': n_annotated,
            'n_flagged': n_flagged,
            'n_exact': n_exact,
            'n_same_type': n_same_type,
            'pct_exact': n_exact / n_annotated * 100 if n_annotated else float('nan'),
            'pct_same_type': n_same_type / n_annotated * 100 if n_annotated else float('nan'),
            'alpha': alpha,
        }

    # The former Disagrees file holds MedExpert Criteria-Checklist rows that normalise
    # to FP/FN, so it is merged into the FP and FN pools rather than reported apart.
    _fp_rows = _load_fpfn(_FP_PATH, _DIS_PATH, 'FP')
    _fn_rows = _load_fpfn(_FN_PATH, _DIS_PATH, 'FN')
    _tp_rows = load_validation_rows(_TP_PATH)

    _fp_stats = _stats_for_file(_fp_rows)
    _fn_stats = _stats_for_file(_fn_rows)

    # Reasoning-alignment rows use human_omissions_aligned rather than human_category.
    def _align_stats(rows, cat):
        subset = [r for r in rows if r.get('analysis_category') == cat]
        annotated = [r for r in subset if str(r.get('human_omissions_aligned') or '').strip()]
        n_total = len(subset)
        n_annotated = len(annotated)
        n_flagged = sum(1 for r in annotated if r.get('human_flagged_for_review') in (True, 'True', 'true', '1', 1))
        n_exact = sum(1 for r in annotated if r.get('assigned_class', '').strip() == str(r.get('human_omissions_aligned', '')).strip())
        alpha = _compute_alpha(subset, 'assigned_class', 'human_omissions_aligned')
        return {'n_total': n_total, 'n_annotated': n_annotated, 'n_flagged': n_flagged, 'n_exact': n_exact, 'alpha': alpha}

    _tp_stats = _align_stats(_tp_rows, 'TP')

    def _fmt_align_row(label, s):
        na = s['n_annotated']
        return {
            'Category': label,
            'Task': 'Reasoning alignment (Yes/Partially/No)',
            'N annotated / N total': f"{na}/{s['n_total']}",
            'Exact agree': f"{s['n_exact']}/{na} ({s['n_exact']/na*100:.0f}%)" if na else 'n/a',
            'Same-type agree': '—',
            'Kripp. α': f"{s['alpha']:.3f}" if not np.isnan(s['alpha']) else 'n/a',
            'N flagged': s['n_flagged'],
        }

    # Summary display
    _summary_rows = [
        _fmt_align_row('TP / agreement-incomplete (in-dataset)', _tp_stats),
        {
            'Category': 'FP (in-dataset)',
            'Task': 'Failure mode sub-code',
            'N annotated / N total': f"{_fp_stats['n_annotated']}/{_fp_stats['n_total']}",
            'Exact agree': f"{_fp_stats['n_exact']}/{_fp_stats['n_annotated']} ({_fp_stats['pct_exact']:.0f}%)",
            'Same-type agree': f"{_fp_stats['n_same_type']}/{_fp_stats['n_annotated']} ({_fp_stats['pct_same_type']:.0f}%)",
            'Kripp. α': f"{_fp_stats['alpha']:.3f}" if not np.isnan(_fp_stats['alpha']) else 'n/a',
            'N flagged': _fp_stats['n_flagged'],
        },
        {
            'Category': 'FN (in-dataset)',
            'Task': 'Failure mode sub-code',
            'N annotated / N total': f"{_fn_stats['n_annotated']}/{_fn_stats['n_total']}",
            'Exact agree': f"{_fn_stats['n_exact']}/{_fn_stats['n_annotated']} ({_fn_stats['pct_exact']:.0f}%)",
            'Same-type agree': f"{_fn_stats['n_same_type']}/{_fn_stats['n_annotated']} ({_fn_stats['pct_same_type']:.0f}%)",
            'Kripp. α': f"{_fn_stats['alpha']:.3f}" if not np.isnan(_fn_stats['alpha']) else 'n/a',
            'N flagged': _fn_stats['n_flagged'],
        },
    ]
    _summary_df = pd.DataFrame(_summary_rows)

    mo.vstack([
        mo.md("### TP / FP / FN validation summary"),
        mo.as_html(_summary_df),
    ])
    return


@app.cell
def _(OUTPUT_DIR, load_validation_rows, np, os, pd, simpledorff):

    _ANNOTATED_DIR2 = os.path.join(OUTPUT_DIR, 'qualitative_analysis', 'annotated')
    _FP_PATH2 = os.path.join(_ANNOTATED_DIR2, 'rq2_qualitative_validation_fp_resampled_annotated.jsonl')
    _FN_PATH2 = os.path.join(_ANNOTATED_DIR2, 'rq2_qualitative_validation_fn_resampled_annotated.jsonl')
    _DIS_PATH2 = os.path.join(_ANNOTATED_DIR2, 'rq2_qualitative_validation_disagrees_resampled_annotated.jsonl')
    _TP_PATH2 = os.path.join(_ANNOTATED_DIR2, 'rq2_qualitative_validation_tp_agrees_resampled_annotated.jsonl')

    def _load2(*paths, category=None):
        rows = [r for path in paths for r in load_validation_rows(path)]
        return [r for r in rows if category is None or r['analysis_category'] == category]

    # The former Disagrees file contributes MedExpert Criteria-Checklist rows that
    # normalise to FP/FN, so it folds into the FP and FN pools.
    _fp2 = _load2(_FP_PATH2, _DIS_PATH2, category='FP')
    _fn2 = _load2(_FN_PATH2, _DIS_PATH2, category='FN')

    # FP05 is absent from the classifier's schema (see the taxonomy heatmap cell);
    # report it as Other. This is a rename, not a merge: 'Other' collides with no
    # human label, so the agreement counts and Krippendorff's alpha are unchanged.
    for _r2 in _fp2:
        if _r2.get('assigned_class') == 'FP05':
            _r2['assigned_class'] = 'Other'
    _tp2 = _load2(_TP_PATH2)

    # --- Per-row detail table for supplement ---
    def _detail_rows(rows, label):
        out = []
        for r in rows:
            hcat = r.get('human_category', '').strip()
            if not hcat:
                continue
            acat = r.get('assigned_class', '').strip()
            match = 'Yes' if acat == hcat else ('Same-type' if (len(acat) >= 2 and len(hcat) >= 2 and acat[:2] == hcat[:2]) else 'No')
            out.append({
                'Category': label,
                'ID': r.get('ID') or r.get('id', ''),
                'LLM label': acat,
                'Human label': hcat,
                'Match': match,
                'Flagged': 'Yes' if r.get('human_flagged_for_review') in (True, 'True', 'true', '1', 1) else '',
            })
        return out

    _tp2_annotated = [r for r in _tp2 if str(r.get('human_omissions_aligned') or '').strip()]
    _tp_detail = []
    for _r in _tp2_annotated:
        _acat = _r.get('assigned_class', '').strip()
        _hcat = str(_r.get('human_omissions_aligned', '')).strip()
        _match = 'Yes' if _acat == _hcat else 'No'
        _tp_detail.append({
            'Category': 'TP',
            'ID': _r.get('ID') or _r.get('id', ''),
            'LLM label': _acat,
            'Human label': _hcat,
            'Match': _match,
            'Flagged': 'Yes' if _r.get('human_flagged_for_review') in (True, 'True', 'true', '1', 1) else '',
        })
    _detail = _tp_detail + _detail_rows(_fp2, 'FP') + _detail_rows(_fn2, 'FN')
    _detail_df2 = pd.DataFrame(_detail)

    # Reload stats (need them here — recompute inline to keep cell self-contained)
    def _same_type2(a, b):
        return isinstance(a, str) and isinstance(b, str) and len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]

    def _quick_stats(rows, cat_filter=None):
        if cat_filter:
            rows = [r for r in rows if r.get('analysis_category') == cat_filter]
        ann = [r for r in rows if r.get('human_category', '').strip()]
        n_tot = len(rows)
        n_ann = len(ann)
        n_flag = sum(1 for r in ann if r.get('human_flagged_for_review') in (True, 'True', 'true', '1', 1))
        n_ex = sum(1 for r in ann if r.get('assigned_class', '').strip() == r.get('human_category', '').strip())
        n_st = sum(1 for r in ann if _same_type2(r.get('assigned_class', ''), r.get('human_category', '')))
        return n_tot, n_ann, n_flag, n_ex, n_st

    def _align_stats2(rows, cat):
        subset = [r for r in rows if r.get('analysis_category') == cat]
        ann = [r for r in subset if str(r.get('human_omissions_aligned') or '').strip()]
        n_tot = len(subset)
        n_ann = len(ann)
        n_flag = sum(1 for r in ann if r.get('human_flagged_for_review') in (True, 'True', 'true', '1', 1))
        n_ex = sum(1 for r in ann if r.get('assigned_class', '').strip() == str(r.get('human_omissions_aligned', '')).strip())
        return n_tot, n_ann, n_flag, n_ex

    _tp_nt2, _tp_na2, _tp_nf2, _tp_ne2 = _align_stats2(_tp2, 'TP')

    def _alpha2_align(rows, cat):
        subset = [r for r in rows if r.get('analysis_category') == cat]
        recs = []
        for i, r in enumerate(subset):
            iid = r.get('ID') or r.get('id') or str(i)
            recs.append({'id': iid, 'annotator': 'llm', 'label': r.get('assigned_class')})
            h = str(r.get('human_omissions_aligned') or '').strip()
            recs.append({'id': iid, 'annotator': 'human', 'label': h if h else None})
        try:
            return simpledorff.calculate_krippendorffs_alpha_for_df(
                pd.DataFrame(recs), experiment_col='id', annotator_col='annotator', class_col='label'
            )
        except Exception:
            return float('nan')

    _tp_alpha2 = _alpha2_align(_tp2, 'TP')

    _fp_nt, _fp_na, _fp_nf, _fp_ne, _fp_ns = _quick_stats(_fp2)
    _fn_nt, _fn_na, _fn_nf, _fn_ne, _fn_ns = _quick_stats(_fn2)

    _val_table_path = os.path.join(OUTPUT_DIR, 'rq2_taxonomy_validation.tex')
    _val_detail_path = os.path.join(OUTPUT_DIR, 'rq2_taxonomy_validation_detail.tex')

    def _alpha2(rows, ac='assigned_class', hc='human_category'):
        recs = []
        for i, r in enumerate(rows):
            iid = r.get('ID') or r.get('id') or str(i)
            recs.append({'id': iid, 'annotator': 'llm', 'label': r.get(ac)})
            h = r.get(hc, '').strip()
            recs.append({'id': iid, 'annotator': 'human', 'label': h if h else None})
        try:
            return simpledorff.calculate_krippendorffs_alpha_for_df(
                pd.DataFrame(recs), experiment_col='id', annotator_col='annotator', class_col='label'
            )
        except Exception:
            return float('nan')

    _fp_alpha2 = _alpha2(_fp2)
    _fn_alpha2 = _alpha2(_fn2)

    def _fmt_alpha(v):
        return f'{v:.3f}' if not (isinstance(v, float) and np.isnan(v)) else '---'

    _tex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\begin{tabular}{llccc}',
        r'    \toprule',
        r'    \textbf{Category} & \textbf{Task} & \textbf{N ann.} & \textbf{Raw Agr.} & $\boldsymbol{\alpha}$ \\',
        r'    \midrule',
        # The 16-item unblinded TP row is superseded by the blinded validation
        # (rq2_blind_validation.tex), so only the FP/FN failure-mode rows are reported.
        f'    FP & Failure mode sub-code & {_fp_nt} & {_fp_ne}/{_fp_na} ({_fp_ne/_fp_na*100:.0f}\\%) & {_fmt_alpha(_fp_alpha2)} \\\\',
        f'    FN & Failure mode sub-code & {_fn_nt} & {_fn_ne}/{_fn_na} ({_fn_ne/_fn_na*100:.0f}\\%) & {_fmt_alpha(_fn_alpha2)} \\\\',
        r'    \bottomrule',
        r'\end{tabular}',
        r'\caption{Human validation of automated taxonomy labels. '
        r'\textit{Exact} = human annotator assigned the identical sub-code as the automated classifier. '
        r'$\alpha$ = Krippendorff\textquotesingle s $\alpha$ (nominal, automated vs.\ human). '
        r'Both completeness definitions are pooled: each row covers the Omission-Harm Rubric and Criteria-Checklist rows alike. '
        r'Annotators saw the classifier\textquotesingle s code. Reasoning-alignment labels were validated separately '
        r'with a blinded annotator (\Cref{tab:rq2_blind_validation}). '
        + (f'Flagged for review: FP {_fp_nf}, FN {_fn_nf} (included in all counts).' if any([_fp_nf, _fn_nf]) else '')
        + r'}\label{tab:taxonomy_validation}',
        r'\end{table}',
    ]

    with open(_val_table_path, 'w') as _f:
        _f.write('\n'.join(_tex_lines) + '\n')
    print(f'Saved {_val_table_path}')

    # --- Supplement detail table ---
    _det_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{llllc}',
        r'    \toprule',
        r'    \textbf{Cat.} & \textbf{LLM label} & \textbf{Human label} & \textbf{Match} & \textbf{Flagged} \\',
        r'    \midrule',
    ]
    for _, _row in _detail_df2.iterrows():
        _cells = [_row['Category'], _row['LLM label'], _row['Human label'], _row['Match'], _row['Flagged']]
        _det_lines.append('    ' + ' & '.join(str(c) for c in _cells) + r' \\')
    _det_lines += [
        r'    \bottomrule',
        r'\end{tabular}',
        r'\caption{Per-annotation detail for human validation of LLM-assigned taxonomy labels.}\label{tab:taxonomy_validation_detail}',
        r'\end{table}',
    ]

    with open(_val_detail_path, 'w') as _f:
        _f.write('\n'.join(_det_lines) + '\n')
    print(f'Saved {_val_detail_path}')
    return


@app.cell
def _(OUTPUT_DIR, load_validation_rows, mo, np, os, pd, plt, sns):
    _ANNOTATED_DIR3 = os.path.join(OUTPUT_DIR, 'qualitative_analysis', 'annotated')

    def _load3(name):
        path = os.path.join(_ANNOTATED_DIR3, name)
        return load_validation_rows(path) if os.path.exists(path) else []

    # The former Disagrees file contributes MedExpert Criteria-Checklist rows that
    # normalise to FP/FN, so it is read alongside the FP and FN files.
    _FM_FILES = [
        'rq2_qualitative_validation_fp_resampled_annotated.jsonl',
        'rq2_qualitative_validation_fn_evened_annotated.jsonl',
        'rq2_qualitative_validation_disagrees_resampled_annotated.jsonl',
    ]

    _records3 = []
    for _fname3 in _FM_FILES:
        for _r in _load3(_fname3):
            _hc3 = str(_r.get('human_category') or '').strip()
            # DIS* codes belong to the retired framework-difference vocabulary.
            if _hc3 and not _hc3.startswith('DIS') and _r['analysis_category'] in ('FP', 'FN'):
                _records3.append({
                    'group': _r['analysis_category'],
                    'llm': _r['assigned_class'],
                    'human': _hc3,
                })
    for _r in _load3('rq2_qualitative_validation_tp_agrees_evened_annotated.jsonl'):
        if _r['analysis_category'] == 'TP' and str(_r.get('human_omissions_aligned') or '').strip():
            _records3.append({'group': 'TP', 'llm': _r['assigned_class'], 'human': _r['human_omissions_aligned']})

    _df3 = pd.DataFrame(_records3)
    _groups3 = [('FP', 'FP'), ('FN', 'FN'), ('TP', 'TP')]
    _groups3 = [_g for _g in _groups3 if _g[0] in _df3['group'].unique()] if not _df3.empty else []

    if not _groups3:
        mo.callout(mo.md('No annotation data available for label-change plot.'), kind='warn')
    else:
        _fig3, _axes3 = plt.subplots(1, len(_groups3), figsize=(4.5 * len(_groups3), 5))
        if len(_groups3) == 1:
            _axes3 = [_axes3]

        for _ax3, (_gkey, _glabel) in zip(_axes3, _groups3):
            _sub3 = _df3[_df3['group'] == _gkey]
            _labels3 = sorted(set(_sub3['llm']) | set(_sub3['human']))
            _ct3 = pd.crosstab(_sub3['llm'], _sub3['human']).reindex(
                index=_labels3, columns=_labels3, fill_value=0)
            _rs3 = _ct3.sum(axis=1).replace(0, np.nan)
            _pct3 = (_ct3.div(_rs3, axis=0) * 100).round(1)
            _annot3 = _pct3.copy().astype(object)
            for _i3 in _pct3.index:
                for _j3 in _pct3.columns:
                    _v3, _n3 = _pct3.loc[_i3, _j3], _ct3.loc[_i3, _j3]
                    _annot3.loc[_i3, _j3] = f'{_n3}\n({_v3:.0f}%)' if _n3 > 0 else ''
            sns.heatmap(
                _pct3, ax=_ax3, annot=_annot3, fmt='', mask=(_pct3 == 0),
                cmap='Blues', vmin=0, vmax=100, linewidths=0.5, linecolor='#ddd',
                cbar_kws={'label': '%', 'shrink': 0.7}, annot_kws={'size': 8},
            )
            for _k3, _lbl3 in enumerate(_pct3.index):
                if _lbl3 in _pct3.columns:
                    _ax3.add_patch(plt.Rectangle(
                        (list(_pct3.columns).index(_lbl3), _k3), 1, 1,
                        fill=False, edgecolor='#e63946', lw=2))
            _ax3.set_title(f'{_glabel}  (n={len(_sub3)})', fontsize=11, fontweight='bold')
            _ax3.set_xlabel('Human label', fontsize=9)
            _ax3.set_ylabel('LLM label', fontsize=9)
            _ax3.tick_params(axis='x', rotation=45, labelsize=8)
            _ax3.tick_params(axis='y', rotation=0, labelsize=8)

        _fig3.suptitle(
            'LLM-assigned code → Human-assigned code  (row = LLM label, % per row)',
            fontsize=11, y=1.01)
        plt.tight_layout()
        _plot_path3 = os.path.join(OUTPUT_DIR, 'rq2_taxonomy_label_changes.pdf')
        _fig3.savefig(_plot_path3, bbox_inches='tight')
        plt.close(_fig3)
        mo.vstack([
            mo.md('### Label change heatmap (LLM → Human)'),
            mo.image(_plot_path3),
        ])
    return


@app.cell
def _(
    OUTPUT_DIR,
    load_validation_rows,
    matplotlib,
    mo,
    mpatches,
    np,
    os,
    pd,
    plt,
    save_figure,
    sns,
):

    _ANN_DIR4 = os.path.join(OUTPUT_DIR, 'qualitative_analysis', 'annotated')

    def _load4(paths):
        rows = {}
        for p in paths:
            if not os.path.exists(p):
                continue
            for r in load_validation_rows(p):
                # Key on the full (response, definition, backbone) triple: the same response
                # appears once per backbone/definition, so keying on ID alone dropped rows.
                rid = (r.get('ID') or r.get('id'), r.get('Metric'), r.get('Eval Model'))
                lbl_col = 'human_omissions_aligned' if r['analysis_category'] in ('TP', 'TN') else 'human_category'
                if rid not in rows or (r.get(lbl_col) not in (None, '') and rows[rid].get(lbl_col) in (None, '')):
                    rows[rid] = r
        return list(rows.values())

    # Same files as the alpha table (rq2_taxonomy_validation.tex), so the heatmap and the
    # reported alphas describe the same rows. The former Disagrees file holds MedExpert
    # Criteria-Checklist rows that normalise to FP/FN, so it is read alongside the FP and FN files.
    _FM_PATHS4 = [os.path.join(_ANN_DIR4, _n) for _n in (
        'rq2_qualitative_validation_fp_resampled_annotated.jsonl',
        'rq2_qualitative_validation_fn_resampled_annotated.jsonl',
        'rq2_qualitative_validation_disagrees_resampled_annotated.jsonl',
    )]
    _fm4 = [
        r for r in _load4(_FM_PATHS4)
        # DIS* codes belong to the retired framework-difference vocabulary.
        if str(r.get('human_category') or '').strip()
        and not str(r['human_category']).startswith('DIS')
    ]
    _fp4 = [r for r in _fm4 if r['analysis_category'] == 'FP']
    _fn4 = [r for r in _fm4 if r['analysis_category'] == 'FN']

    _tp4 = [r for r in _load4([
        os.path.join(_ANN_DIR4, 'rq2_qualitative_validation_tp_agrees_resampled_annotated.jsonl'),
    ]) if r['analysis_category'] == 'TP' and r.get('human_omissions_aligned') not in (None, '')]

    def _pivot4(rows, llm_col, human_col, llm_order, human_order=None):
        _df = pd.DataFrame([{'llm': r.get(llm_col), 'human': r.get(human_col)} for r in rows])
        _ct = pd.crosstab(_df['llm'], _df['human'])
        _ct = _ct.reindex([c for c in llm_order if c in _ct.index], axis=0)
        if human_order:
            _ct = _ct.reindex([c for c in human_order if c in _ct.columns], axis=1, fill_value=0)
        _ct = _ct.fillna(0).astype(int)
        _pct = _ct.div(_ct.sum(axis=1).replace(0, np.nan), axis=0).multiply(100).fillna(0)
        return _pct, _ct

    # FP05 is not in the classifier's schema: prompt2_different_verdict.md offers
    # only FP01-FP04 / FN01-FN05, and the judge coined FP05 on its own. The
    # annotation UI defines FP05 as "Other", so report it that way rather than as
    # a taxonomy category of its own.
    for _r4 in _fp4:
        if _r4.get('assigned_class') == 'FP05':
            _r4['assigned_class'] = 'Other'

    _fp_pct4, _fp_ct4 = _pivot4(_fp4, 'assigned_class', 'human_category',
        llm_order=['FP01', 'FP02', 'FP03', 'FP04', 'Other'],
        human_order=['FP01', 'FP02', 'FP03', 'FP04'])
    _fn_pct4, _fn_ct4 = _pivot4(_fn4, 'assigned_class', 'human_category',
        llm_order=['FN01', 'FN02', 'FN03', 'FN04', 'FN05'])
    _tp_pct4, _tp_ct4 = _pivot4(_tp4, 'assigned_class', 'human_omissions_aligned',
        llm_order=['Yes', 'Partially', 'No'],
        human_order=['Yes', 'Partially', 'No'])

    def _draw4(ax, pct, counts, title, show_ylabel=True):
        _annot = pd.DataFrame(
            [[f'{int(round(pct.iloc[i, j]))}%\n({int(counts.iloc[i, j])})' for j in range(pct.shape[1])]
             for i in range(pct.shape[0])],
            index=pct.index, columns=pct.columns)
        sns.heatmap(pct, ax=ax, annot=_annot, fmt='', cmap='Blues', vmin=0, vmax=100,
            linewidths=0.5, linecolor='white', cbar=False,
            annot_kws={'fontsize': 12})
        for _i, _rl in enumerate(pct.index):
            if _rl in list(pct.columns):
                _j = list(pct.columns).index(_rl)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (_j, _i), 1, 1, boxstyle='square,pad=0',
                    linewidth=1.5, edgecolor='#2ca02c', facecolor='#b7e4b7', zorder=0))
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Human Label', fontsize=9)
        ax.set_ylabel('LLM Label' if show_ylabel else '', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    # The unblinded TP panel is dropped: the blinded validation (rq2_blind_confusion.tex)
    # replaces it, so the figure shows the FP/FN failure-mode codes only.
    if not _fp4 and not _fn4:
        mo.callout(mo.md('No annotation data available for taxonomy validation heatmap.'), kind='warn')
    else:
        _fig4, _axes4 = plt.subplots(1, 2, figsize=(12, 5))
        _draw4(_axes4[0], _fp_pct4, _fp_ct4, 'FP: Over-penalization', show_ylabel=True)
        _draw4(_axes4[1], _fn_pct4, _fn_ct4, 'FN: Missed omission', show_ylabel=False)
        _fig4.suptitle('Per-code Agreement: LLM vs Human Labels', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        _sm4 = plt.cm.ScalarMappable(cmap='Blues', norm=matplotlib.colors.Normalize(vmin=0, vmax=100))
        _sm4.set_array([])
        _cbar4 = _fig4.colorbar(_sm4, ax=_axes4, shrink=0.8, pad=0.02)
        _cbar4.set_label('Row %', fontsize=9)
        save_figure(_fig4, os.path.join(OUTPUT_DIR, 'rq2_taxonomy_validation_heatmap'))
        plt.close(_fig4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Supplementary Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Malformed Output Rates

    Percentage of records with at least one malformed (unparseable) criterion response, broken down by model, rubric, and dataset.
    When JSON parsing fails the criterion is auto-assigned the minimum score (mapping to Incomplete).
    """)
    return


@app.cell
def _(OUTPUT_DIR, SETTINGS, completeness_methods, mo, os, pd, results_df_all):
    _METRIC_DISPLAY = {k: v for k, v in completeness_methods.items() if k != 'completeness-human'}
    _MODEL_ORDER = [s['model'] for s in SETTINGS]
    _DATASET_ORDER = ['MedExpert', 'HealthBench', 'HealthBench-MT']

    def _has_parse_error(metric_details):
        if not isinstance(metric_details, dict):
            return False
        for crit in (metric_details.get('meta') or []):
            expl = (crit.get('explanation') or '') if isinstance(crit, dict) else ''
            if 'Error' in expl or 'error' in expl:
                return True
        return False

    _df = results_df_all.reset_index().copy()
    _df = _df[_df['metric'].isin(_METRIC_DISPLAY)].copy()
    _df['_has_error'] = _df['metric_details'].apply(_has_parse_error)

    _error_counts = (
        _df[_df['_has_error']]
        .groupby(['eval_model', 'dataset', 'metric'])['id']
        .nunique()
        .reset_index(name='n_errored')
    )
    _total_counts = (
        _df.groupby(['eval_model', 'dataset', 'metric'])['id']
        .nunique()
        .reset_index(name='n_total')
    )
    _merged = _total_counts.merge(_error_counts, on=['eval_model', 'dataset', 'metric'], how='left')
    _merged['n_errored'] = _merged['n_errored'].fillna(0).astype(int)
    _merged['pct'] = (_merged['n_errored'] / _merged['n_total'] * 100).round(1)
    _merged['cell'] = _merged.apply(
        lambda r: f"{r['pct']:.1f}" if r['n_errored'] > 0 else '0', axis=1
    )
    _merged['metric_display'] = _merged['metric'].map(_METRIC_DISPLAY)

    # Pivot: rows = (model, metric), columns = dataset
    _datasets_present = [d for d in _DATASET_ORDER if d in _merged['dataset'].unique()]
    _pivot = (
        _merged
        .pivot_table(
            index=['eval_model', 'metric_display'],
            columns='dataset',
            values='cell',
            aggfunc='first',
        )
        .reindex(columns=_datasets_present)
        .fillna('--')
    )

    # Reorder rows by SETTINGS model order, then completeness_methods metric order
    _metric_order = list(_METRIC_DISPLAY.values())
    _index_tuples = [
        (m, met)
        for m in _MODEL_ORDER
        for met in _metric_order
        if (m, met) in _pivot.index
    ]
    _pivot = _pivot.reindex(
        pd.MultiIndex.from_tuples(_index_tuples, names=['eval_model', 'metric_display'])
    ).dropna(how='all')

    malformed_rate_df = _pivot.reset_index().rename(
        columns={'eval_model': 'Model', 'metric_display': 'Metric'}
    )

    # ── Save LaTeX ──────────────────────────────────────────────────────────────
    _col_spec = 'll' + 'r' * len(_datasets_present)
    _tex_rows = [
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{' + _col_spec + r'}',
        r'\toprule',
        r'\textbf{Model} & \textbf{Metric} & ' + ' & '.join([r'\textbf{' + d + r'}' for d in _datasets_present]) + r' \\',
        r'\midrule',
    ]
    _prev_model = None
    for (_model, _metric) in _pivot.index:
        if _model != _prev_model and _prev_model is not None:
            _tex_rows.append(r'    \midrule')
        _model_cell = _model if _model != _prev_model else ''
        _prev_model = _model
        _vals = [str(_pivot.loc[(_model, _metric), d]) for d in _datasets_present]
        _tex_rows.append(f'    {_model_cell} & {_metric} & ' + ' & '.join(_vals) + r' \\')
    _tex_rows += [
        r'    \bottomrule',
        r'    \end{tabular}',
        (
            r'\caption{Percentage of records (\%) with at least one malformed (unparseable) criterion output per model, rubric, and dataset. Zero indicates no malformed outputs; \texttt{--} indicates the combination was not evaluated. '
            r'Malformed outputs are auto-assigned the minimum score (Incomplete).'
            r'Prometheus-2~7B failures arise from context-length truncation '
            r'(\texttt{finish\_reason=length}).}'
        ),
        r'\label{tab:malformed_output_rates}',
        r'\end{table}',
    ]
    _out_path = os.path.join(OUTPUT_DIR, 'malformed_output_rates.tex')
    with open(_out_path, 'w') as _f:
        _f.write('\n'.join(_tex_rows) + '\n')
    print(f'Saved {_out_path}')

    mo.ui.table(malformed_rate_df, selection=None, page_size=len(malformed_rate_df))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ML4H 2026 Rebuttal Analyses

    Re-analyses requested by the ML4H 2026 (submission 210) reviewers; see
    `ML4H_2026/review_points.md` for the point numbers (§) cited in each cell. Outputs go to
    `tables_media/ml4h_rebuttal/` and do not replace the paper tables. All cells use the
    reported MedExpert pairings (Omission-Harm Rubric FS and Criteria-Checklist) and add
    OpenBioLLM-70B alongside the four main-text backbones.
    """)
    return


@app.cell
def _(OUTPUT_DIR, completeness_accuracy_df, os, question_cluster_id):
    REBUTTAL_DIR = os.path.join(OUTPUT_DIR, 'ml4h_rebuttal')
    os.makedirs(REBUTTAL_DIR, exist_ok=True)
    REBUTTAL_OH = 'Omission-Harm Rubric'
    REBUTTAL_CC = 'Criteria-Checklist'
    SHARED_BACKBONES = ['LLaMA 3.3-70B', 'GPT-5 Mini', 'Qwen3.6']
    REBUTTAL_BACKBONES = SHARED_BACKBONES + ['Prometheus-2 7B', 'OpenBioLLM']

    # Same rows as Table 1: the two reported MedExpert pairings, every backbone.
    rebuttal_df = completeness_accuracy_df[
        (completeness_accuracy_df['dataset'] == 'MedExpert')
        & (completeness_accuracy_df['pairing_type'] == 'in-dataset')
        & completeness_accuracy_df['eval_model'].isin(REBUTTAL_BACKBONES)
    ].copy()
    assert rebuttal_df[['human_judgment', 'model_judgment', 'score']].notna().all().all()
    rebuttal_df['title'] = rebuttal_df['meta'].apply(lambda m: m['title'])
    rebuttal_df['qid'] = rebuttal_df['title'].apply(question_cluster_id)
    rebuttal_df['domain'] = rebuttal_df['meta'].apply(lambda m: m['domain'])
    rebuttal_df['generator'] = rebuttal_df['meta'].apply(lambda m: m['model'])
    assert rebuttal_df['qid'].notna().all(), 'unparseable MedExpert title'
    _n_questions = rebuttal_df['qid'].nunique()
    assert _n_questions == 108, f'expected 108 question clusters, got {_n_questions}'
    ID_TO_QID = dict(zip(rebuttal_df['id'], rebuttal_df['qid']))
    ID_TO_DOMAIN = dict(zip(rebuttal_df['id'], rebuttal_df['domain']))
    ID_TO_GENERATOR = dict(zip(rebuttal_df['id'], rebuttal_df['generator']))
    print(
        f'Rebuttal frame: {len(rebuttal_df)} rows, {rebuttal_df["id"].nunique()} responses, '
        f'{_n_questions} questions'
    )
    return (
        ID_TO_DOMAIN,
        ID_TO_GENERATOR,
        ID_TO_QID,
        REBUTTAL_BACKBONES,
        REBUTTAL_CC,
        REBUTTAL_DIR,
        REBUTTAL_OH,
        SHARED_BACKBONES,
        rebuttal_df,
    )


@app.cell
def _(average_precision_score, np, roc_auc_score):
    def kappa_alpha_binary(rater_a, rater_b):
        """Cohen's kappa and Krippendorff's alpha (nominal) for two raters, boolean arrays.

        Closed forms for two raters with no missing values, so they are cheap enough to
        recompute inside a bootstrap. Kappa uses each rater's own marginals; alpha pools
        them, which is why alpha goes strongly negative when one rater labels nearly
        everything Incomplete while kappa stays near zero.
        """
        a = np.asarray(rater_a, dtype=bool)
        b = np.asarray(rater_b, dtype=bool)
        n = len(a)
        if n == 0:
            return np.nan, np.nan
        disagree = np.count_nonzero(a != b)
        p_o = 1 - disagree / n
        p_a, p_b = a.mean(), b.mean()
        p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan
        n_true = np.count_nonzero(a) + np.count_nonzero(b)
        n_false = 2 * n - n_true
        alpha = 1 - (2 * n - 1) * disagree / (n_true * n_false) if n_true and n_false else np.nan
        return kappa, alpha

    def discrimination_metrics(y_true_incomplete, y_pred_incomplete, scores):
        """Table 1 statistics plus AUROC and Cohen's kappa. Incomplete is the positive class.

        `scores` run high = more complete, so they are negated for AUC-PR/AUROC. AUROC is
        threshold-free and prevalence-independent; AUC-PR is reported in the paper because
        its random baseline equals the Incomplete prevalence (0.34).
        """
        y = np.asarray(y_true_incomplete, dtype=bool)
        p = np.asarray(y_pred_incomplete, dtype=bool)
        s = -np.asarray(scores, dtype=float)
        tp = np.count_nonzero(y & p)
        fp = np.count_nonzero(~y & p)
        fn = np.count_nonzero(y & ~p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        kappa, alpha = kappa_alpha_binary(y, p)
        both_classes = y.any() and not y.all()
        return {
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': float((y == p).mean()),
            'AUC-PR': average_precision_score(y, s) if both_classes else np.nan,
            'AUROC': roc_auc_score(y, s) if both_classes else np.nan,
            'kappa': kappa,
            'alpha': alpha,
        }

    return discrimination_metrics, kappa_alpha_binary


@app.cell
def _(cohen_kappa_score, kappa_alpha_binary, np, rebuttal_df, simpledorff):
    # Check the closed forms against the library implementations on real data.
    _g = rebuttal_df[rebuttal_df['eval_model'] == 'GPT-5 Mini']
    _g = _g[_g['metric'] == _g['metric'].iloc[0]]
    _y = (_g['human_judgment'] == 'Incomplete').values
    _p = (_g['model_judgment'] == 'Incomplete').values
    _kappa, _alpha = kappa_alpha_binary(_y, _p)
    _melted = _g.melt(id_vars='id', value_vars=['human_judgment', 'model_judgment'], var_name='annotator', value_name='annotation')
    _alpha_ref = simpledorff.calculate_krippendorffs_alpha_for_df(_melted, experiment_col='id', annotator_col='annotator', class_col='annotation')
    assert np.isclose(_kappa, cohen_kappa_score(_y, _p)), (_kappa, cohen_kappa_score(_y, _p))
    assert np.isclose(_alpha, _alpha_ref), (_alpha, _alpha_ref)
    print(f'kappa/alpha closed forms match sklearn/simpledorff (kappa={_kappa:.3f}, alpha={_alpha:.3f})')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §8, §15 — Question-clustered bootstrap CIs, Cohen's κ, and AUROC

    Table 1 CIs resample responses (BCa, 1,000 resamples). Responses to the same question
    share clinical content, so this cell recomputes percentile CIs two ways with 2,000
    resamples: over responses, and over the 108 questions (cluster bootstrap). Cohen's κ and
    AUROC are added next to Krippendorff's α and AUC-PR.
    """)
    return


@app.cell
def _(
    ID_TO_QID,
    REBUTTAL_BACKBONES,
    REBUTTAL_DIR,
    completeness_accuracy_df,
    discrimination_metrics,
    make_cluster_sampler,
    mo,
    np,
    os,
    pd,
    rebuttal_df,
):
    _N_RESAMPLES = 2000
    _STATS = ['F1', 'Precision', 'Recall', 'Accuracy', 'AUC-PR', 'AUROC', 'kappa', 'alpha']

    def _bootstrap_metric_cis(group, n_resamples=_N_RESAMPLES, seed=42):
        """Point estimates plus response-level and question-level 95% percentile CIs."""
        y = (group['human_judgment'] == 'Incomplete').values
        p = (group['model_judgment'] == 'Incomplete').values
        s = group['score'].values
        point = discrimination_metrics(y, p, s)
        samplers = {
            'response': lambda rng, n=len(group): rng.integers(0, n, n),
            'question': make_cluster_sampler(group['qid'].values),
        }
        rows = []
        draws_by_unit = {}
        for unit, sampler in samplers.items():
            rng = np.random.default_rng(seed)
            draws = [discrimination_metrics(y[idx], p[idx], s[idx]) for idx in (sampler(rng) for _ in range(n_resamples))]
            draws_by_unit[unit] = pd.DataFrame(draws)
        for stat in _STATS:
            row = {'stat': stat, 'estimate': point[stat], 'n': len(group), 'n_questions': group['qid'].nunique()}
            for unit, draws in draws_by_unit.items():
                lo, hi = np.nanpercentile(draws[stat].values, [2.5, 97.5])
                row[f'{unit}_lo'], row[f'{unit}_hi'] = lo, hi
            row['width_ratio'] = (row['question_hi'] - row['question_lo']) / (row['response_hi'] - row['response_lo'])
            rows.append(row)
        return pd.DataFrame(rows)

    # The zero-shot Omission-Harm Rubric is not a reported pairing, so it is not in
    # rebuttal_df; it is added here only so the appendix table of every configuration
    # (tab:discrimination_stats_full) gets the same question-clustered CIs.
    _zero_shot = completeness_accuracy_df[
        (completeness_accuracy_df['dataset'] == 'MedExpert')
        & (completeness_accuracy_df['metric'] == 'Omission-Harm Rubric (Zero-shot)')
        & completeness_accuracy_df['eval_model'].isin(REBUTTAL_BACKBONES)
    ].assign(qid=lambda d: d['id'].map(ID_TO_QID))
    assert _zero_shot['qid'].notna().all(), 'zero-shot responses missing from the reported pairings'
    _ci_input = pd.concat([rebuttal_df, _zero_shot], ignore_index=True)

    with mo.persistent_cache(name='rebuttal_cluster_bootstrap_ci'):
        cluster_ci_df = (
            _ci_input.groupby(['metric', 'eval_model'])
            .apply(_bootstrap_metric_cis, include_groups=False)
            .reset_index(level=[0, 1])
            .reset_index(drop=True)
        )
    cluster_ci_df.to_csv(os.path.join(REBUTTAL_DIR, 'cluster_bootstrap_ci.csv'), index=False)

    # Compact wide view: estimate [question-level CI] per stat.
    _wide = cluster_ci_df.assign(
        cell=lambda d: d.apply(lambda r: f"{r['estimate']:.3f} [{r['question_lo']:.3f}, {r['question_hi']:.3f}]", axis=1)
    ).pivot(index=['metric', 'eval_model'], columns='stat', values='cell')[_STATS].reset_index()
    _wide.to_csv(os.path.join(REBUTTAL_DIR, 'cluster_bootstrap_ci_wide.csv'), index=False)
    mo.vstack([
        mo.md('**Estimate [question-clustered 95% CI]**'),
        mo.ui.table(_wide, selection=None, page_size=len(_wide)),
        mo.md('**Width of question-level CI relative to response-level CI**'),
        mo.ui.table(
            cluster_ci_df.pivot(index=['metric', 'eval_model'], columns='stat', values='width_ratio')[_STATS].round(2).reset_index(),
            selection=None,
        ),
    ])
    return (cluster_ci_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §7, §8 — Paired definition comparison (TOST) with question-clustered resampling

    Re-runs `paired_definition_bootstrap` (10,000 resamples, ±2 F1 equivalence margin)
    resampling questions instead of responses, side by side with the response-level result.
    """)
    return


@app.cell
def _(REBUTTAL_DIR, mo, os, paired_definition_bootstrap, paired_definition_df, pd, rebuttal_df):
    with mo.persistent_cache(name='rebuttal_paired_definition_bootstrap_question'):
        _clustered = paired_definition_bootstrap(rebuttal_df, cluster_col='qid')
    _keep = ['eval_model', 'metric', 'n', 'delta', 'ci90_lo', 'ci90_hi', 'ci95_lo', 'ci95_hi', 'p_two_sided', 'tost_verdict']
    paired_cluster_df = pd.concat([
        paired_definition_df[_keep].assign(resampling_unit='response'),
        _clustered[_keep].assign(resampling_unit='question'),
    ]).sort_values(['metric', 'eval_model', 'resampling_unit']).reset_index(drop=True)
    paired_cluster_df.to_csv(os.path.join(REBUTTAL_DIR, 'paired_definition_bootstrap_clustered.csv'), index=False)

    # The question-clustered run is the one the paper quotes: per-backbone TOST verdicts.
    _tost = _clustered[_keep].copy()
    _tost.to_csv(os.path.join(REBUTTAL_DIR, 'paired_definition_tost.csv'), index=False)
    _fmt_p = lambda p: '$<$0.001' if p < 0.001 else f'{p:.3f}'
    _tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{llrccl}',
        r'    \toprule',
        r'    \textbf{Metric} & \textbf{Judge} & $\boldsymbol{\Delta}$ & \textbf{90\% CI} & \textbf{95\% CI} & \textbf{Verdict} \\',
        r'    \midrule',
    ]
    _metrics = ['F1', 'AUC-PR']
    for _mi, _metric in enumerate(_metrics):
        _sub = _tost[_tost['metric'] == _metric]
        for _i, (_, _r) in enumerate(_sub.iterrows()):
            _label = r'\multirow{' + str(len(_sub)) + r'}{*}{' + _metric + r'}' if _i == 0 else ''
            _judge = _r['eval_model'] + (r'$^*$' if _r['eval_model'] == 'OpenBioLLM' else '')
            _tex.append(
                f"    {_label} & {_judge} & {_r['delta']:+.1f} & [{_r['ci90_lo']:+.1f}, {_r['ci90_hi']:+.1f}] & "
                f"[{_r['ci95_lo']:+.1f}, {_r['ci95_hi']:+.1f}] & {_r['tost_verdict']} \\\\"
            )
        if _mi < len(_metrics) - 1:
            _tex.append(r'    \midrule')
    _margin = float(_clustered['margin'].iloc[0])
    _tex += [
        r'    \bottomrule',
        r'\end{tabular}',
        r'\caption{Paired comparison of the two prompt configurations on the same responses and clinician '
        r'labels: Criteria-Checklist minus Omission-Harm Rubric, in points (scaled by 100). Intervals come '
        r'from a question-clustered paired bootstrap (10{,}000 resamples of the 108 questions). '
        r'\textit{Equivalent}: the 90\% interval lies inside $\pm' + f'{_margin:g}' + r'$ points (two one-sided '
        r'tests); \textit{different}: the 95\% interval excludes zero; \textit{inconclusive}: neither. '
        r'The equivalence margin was set for F1; AUC-PR rows use the same margin for reference. '
        r'$^*$OpenBioLLM-70B is excluded from the main text (\Cref{app:openbiollm}).}'
        r'\label{tab:paired_definition_tost}',
        r'\end{table}',
    ]
    with open(os.path.join(REBUTTAL_DIR, 'paired_definition_tost.tex'), 'w') as _f:
        _f.write('\n'.join(_tex) + '\n')
    mo.ui.table(paired_cluster_df.round(3), selection=None, page_size=len(paired_cluster_df))
    return (paired_cluster_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §1 — Reasoning alignment: shared backbones and matched TP pairs

    (a) Pooled alignment rates per definition, with the pooling set made explicit (the paper's
    15.2% pooled four Omission-Harm backbones, including Prometheus-2 7B, which has no
    Criteria-Checklist run). (b) The **matched** comparison: (response, backbone) pairs that are
    true positives under *both* definitions, so both alignment labels describe the same response
    judged by the same backbone. Uncertainty for the paired difference comes from a
    question-clustered paired bootstrap; McNemar's exact test is on the discordant pairs.
    """)
    return


@app.cell
def _(ID_TO_QID, REBUTTAL_CC, REBUTTAL_OH, full_enriched_df):
    # True-positive pairs (clinician and judge both Incomplete) with their classifier label.
    rebuttal_tp_df = full_enriched_df[
        (full_enriched_df['Dataset'] == 'MedExpert')
        & full_enriched_df['Metric'].isin([REBUTTAL_OH, REBUTTAL_CC])
        & full_enriched_df['Human Judgment'].str.startswith('Incomplete')
        & full_enriched_df['Model Judgment'].str.startswith('Incomplete')
    ].copy()
    rebuttal_tp_df['aligned'] = rebuttal_tp_df['model_response_parsed'].apply(
        lambda x: x.get('omissions_aligned') if isinstance(x, dict) else None
    )
    rebuttal_tp_df['qid'] = rebuttal_tp_df['ID'].map(ID_TO_QID)
    assert rebuttal_tp_df['qid'].notna().all(), 'TP row without a question cluster'
    assert rebuttal_tp_df['aligned'].isin(['Yes', 'Partially', 'No']).all(), rebuttal_tp_df['aligned'].value_counts(dropna=False)
    return (rebuttal_tp_df,)


@app.cell
def _(
    REBUTTAL_BACKBONES,
    REBUTTAL_CC,
    REBUTTAL_DIR,
    REBUTTAL_OH,
    SHARED_BACKBONES,
    make_cluster_sampler,
    mo,
    np,
    os,
    pd,
    rebuttal_tp_df,
    sp,
):
    _ALIGN = ['Yes', 'Partially', 'No']
    _POOLS = {
        **{b: [b] for b in REBUTTAL_BACKBONES},
        'Shared 3 (LLaMA, GPT-5 Mini, Qwen)': SHARED_BACKBONES,
        'Main text 4 (+ Prometheus-2 7B)': SHARED_BACKBONES + ['Prometheus-2 7B'],
        'All 5': REBUTTAL_BACKBONES,
    }

    # (a) Pooled rates, every pooling set made explicit.
    _rows = []
    for _rubric in (REBUTTAL_OH, REBUTTAL_CC):
        for _pool, _members in _POOLS.items():
            _sub = rebuttal_tp_df[(rebuttal_tp_df['Metric'] == _rubric) & rebuttal_tp_df['Eval Model'].isin(_members)]
            if _sub.empty:
                continue
            _counts = _sub['aligned'].value_counts()
            _rows.append({
                'definition': _rubric, 'pool': _pool, 'n_tp': len(_sub),
                **{f'n_{a}': int(_counts.get(a, 0)) for a in _ALIGN},
                **{f'pct_{a}': 100 * _counts.get(a, 0) / len(_sub) for a in _ALIGN},
            })
    alignment_pooled_df = pd.DataFrame(_rows)
    alignment_pooled_df.to_csv(os.path.join(REBUTTAL_DIR, 'alignment_pooled.csv'), index=False)

    # (b) Matched pairs: TP under both definitions for the same response and backbone.
    _cols = ['ID', 'Eval Model', 'qid', 'aligned']
    alignment_matched_df = (
        rebuttal_tp_df[rebuttal_tp_df['Metric'] == REBUTTAL_OH][_cols]
        .merge(
            rebuttal_tp_df[rebuttal_tp_df['Metric'] == REBUTTAL_CC][_cols],
            on=['ID', 'Eval Model', 'qid'], suffixes=('_oh', '_cc'),
        )
    )
    alignment_matched_df.to_csv(os.path.join(REBUTTAL_DIR, 'alignment_matched_pairs_rows.csv'), index=False)

    def _paired_rate_diff(sub, positive, n_resamples=10000, seed=42):
        """CC minus OH rate of `positive` labels on matched pairs, question-clustered bootstrap."""
        x_oh = sub['aligned_oh'].isin(positive).values
        x_cc = sub['aligned_cc'].isin(positive).values
        draw = make_cluster_sampler(sub['qid'].values)
        rng = np.random.default_rng(seed)
        diffs = np.array([x_cc[idx].mean() - x_oh[idx].mean() for idx in (draw(rng) for _ in range(n_resamples))])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        only_oh = int(np.count_nonzero(x_oh & ~x_cc))
        only_cc = int(np.count_nonzero(~x_oh & x_cc))
        p_mcnemar = sp.stats.binomtest(min(only_oh, only_cc), only_oh + only_cc, 0.5).pvalue if only_oh + only_cc else np.nan
        return {
            'rate_oh': 100 * x_oh.mean(), 'rate_cc': 100 * x_cc.mean(),
            'diff_cc_minus_oh': 100 * (x_cc.mean() - x_oh.mean()),
            'diff_ci95_lo': 100 * lo, 'diff_ci95_hi': 100 * hi,
            'discordant_oh_only': only_oh, 'discordant_cc_only': only_cc,
            'mcnemar_exact_p': p_mcnemar,
        }

    _rows = []
    for _pool, _members in _POOLS.items():
        _sub = alignment_matched_df[alignment_matched_df['Eval Model'].isin(_members)]
        if _sub.empty:
            continue
        for _outcome, _positive in (('Yes', ['Yes']), ('Yes or Partially', ['Yes', 'Partially'])):
            _rows.append({
                'pool': _pool, 'outcome': _outcome, 'n_pairs': len(_sub),
                'n_responses': _sub['ID'].nunique(), 'n_questions': _sub['qid'].nunique(),
                **_paired_rate_diff(_sub, _positive),
            })
    alignment_matched_summary_df = pd.DataFrame(_rows)
    alignment_matched_summary_df.to_csv(os.path.join(REBUTTAL_DIR, 'alignment_matched_pairs.csv'), index=False)

    _shared = alignment_matched_df[alignment_matched_df['Eval Model'].isin(SHARED_BACKBONES)]
    alignment_transition_df = pd.crosstab(
        _shared['aligned_oh'].rename('Omission-Harm'), _shared['aligned_cc'].rename('Criteria-Checklist'), margins=True
    ).reindex(index=_ALIGN + ['All'], columns=_ALIGN + ['All'])
    alignment_transition_df.to_csv(os.path.join(REBUTTAL_DIR, 'alignment_matched_transition_shared3.csv'))

    # LaTeX: unpaired TP pools and the matched-pair comparison, shared backbones and each
    # shared backbone. OpenBioLLM and Prometheus-2 7B are left out: the first is excluded
    # from the main text and the second has no Criteria-Checklist run.
    _SHARED_POOL = 'Shared 3 (LLaMA, GPT-5 Mini, Qwen)'
    _fmt_p = lambda p: '$<$0.001' if p < 0.001 else f'{p:.2f}'
    _pooled = alignment_pooled_df.set_index(['definition', 'pool'])
    _tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{lrrrcr}',
        r'    \toprule',
        r'    \textbf{Judge} & $\boldsymbol{n}$ & \textbf{Omission-Harm} & \textbf{Criteria-Checklist} & $\boldsymbol{\Delta}$ \textbf{[95\% CI]} & $\boldsymbol{p}$ \\',
        r'    \midrule',
    ]
    for _outcome, _cols in (('Yes', ['pct_Yes']), ('Yes or Partially', ['pct_Yes', 'pct_Partially'])):
        _title = 'Full alignment (\\textit{Yes})' if _outcome == 'Yes' else 'Full or partial alignment (\\textit{Yes} or \\textit{Partially})'
        _tex.append(r'    \multicolumn{6}{l}{\textit{' + _title + r'}} \\')
        _oh, _cc = _pooled.loc[(REBUTTAL_OH, _SHARED_POOL)], _pooled.loc[(REBUTTAL_CC, _SHARED_POOL)]
        _rate_oh, _rate_cc = sum(_oh[c] for c in _cols), sum(_cc[c] for c in _cols)
        _tex.append(
            f"    Shared, unpaired & {int(_oh['n_tp'])}/{int(_cc['n_tp'])} & {_rate_oh:.1f} & {_rate_cc:.1f} & "
            f"{_rate_cc - _rate_oh:+.1f} & -- \\\\"
        )
        _sub = alignment_matched_summary_df[alignment_matched_summary_df['outcome'] == _outcome].set_index('pool')
        for _pool, _label in [(_SHARED_POOL, 'Shared, matched')] + [(b, '\\quad ' + b) for b in SHARED_BACKBONES]:
            _r = _sub.loc[_pool]
            _tex.append(
                f"    {_label} & {int(_r['n_pairs'])} & {_r['rate_oh']:.1f} & {_r['rate_cc']:.1f} & "
                f"{_r['diff_cc_minus_oh']:+.1f} [{_r['diff_ci95_lo']:+.1f}, {_r['diff_ci95_hi']:+.1f}] & "
                f"{_fmt_p(_r['mcnemar_exact_p'])} \\\\"
            )
        if _outcome == 'Yes':
            _tex.append(r'    \midrule')
    _tex += [
        r'    \bottomrule',
        r'\end{tabular}',
        r'\caption{Reasoning alignment of shared incomplete verdicts (True Positives) under the two prompt '
        r'configurations (Omission-Harm Rubric and Criteria-Checklist), as \% of pairs, on the three '
        r'backbones run with both. \textit{Unpaired}: all TP pairs of each configuration ($n$ = '
        r'Omission-Harm/Criteria-Checklist). '
        r'\textit{Matched}: (response, backbone) pairs that are TP under both configurations, so both labels '
        r'describe the same response judged by the same backbone. $\Delta$ = Criteria-Checklist $-$ Omission-Harm in percentage points, with '
        r'a question-clustered paired bootstrap 95\% CI (10{,}000 resamples); $p$: exact McNemar test on the '
        r'discordant pairs.}\label{tab:alignment_matched_pairs}',
        r'\end{table}',
    ]
    with open(os.path.join(REBUTTAL_DIR, 'alignment_matched_pairs.tex'), 'w') as _f:
        _f.write('\n'.join(_tex) + '\n')

    mo.vstack([
        mo.md('**(a) Pooled TP alignment by pooling set**'),
        mo.ui.table(alignment_pooled_df.round(1), selection=None, page_size=len(alignment_pooled_df)),
        mo.md('**(b) Matched TP pairs: Criteria-Checklist minus Omission-Harm (question-clustered 95% CI)**'),
        mo.ui.table(alignment_matched_summary_df.round(3), selection=None, page_size=len(alignment_matched_summary_df)),
        mo.md('**Label transitions on matched pairs, shared three backbones**'),
        mo.ui.table(alignment_transition_df.reset_index(), selection=None),
    ])
    return (
        alignment_matched_df,
        alignment_matched_summary_df,
        alignment_pooled_df,
        alignment_transition_df,
    )


@app.cell
def _(RQ2_ANALYSIS_RESULTS_PATH, REBUTTAL_CC, REBUTTAL_DIR, mo, os, pd):
    # Issue (i): which category the classifier prompt received for Criteria-Checklist TP rows.
    # Rows built under the retired in-/cross-dataset split still carry 'Agrees', which
    # prompt1_same_verdict.md never defines; Omission-Harm rows received 'TP'. The rerun is
    # deferred, so this only records the scope.
    _res = pd.read_json(RQ2_ANALYSIS_RESULTS_PATH, lines=True)
    _res['Metric'] = _res['Metric'].replace({'Dynamic-Checklist': REBUTTAL_CC})
    _tp_cc = _res[
        (_res['Dataset'] == 'MedExpert') & (_res['Metric'] == REBUTTAL_CC)
        & _res['Human Judgment'].str.startswith('Incomplete') & _res['Model Judgment'].str.startswith('Incomplete')
    ]
    cc_prompt_category_df = pd.crosstab(_tp_cc['Eval Model'], _tp_cc['analysis_category'], margins=True)
    cc_prompt_category_df.to_csv(os.path.join(REBUTTAL_DIR, 'cc_tp_prompt_category.csv'))
    mo.vstack([
        mo.md('**Criteria-Checklist TP rows by the category passed to the classifier prompt**'),
        mo.ui.table(cc_prompt_category_df.reset_index(), selection=None),
    ])
    return (cc_prompt_category_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §9 — Per-domain and per-generator breakdown (rpj2 Q4)

    Verdict-level statistics and TP alignment rates separately for mental health (MH) vs.
    prenatal care (PC), and for each chatbot that generated the responses. CIs resample
    questions within the stratum (1,000 resamples).
    """)
    return


@app.cell
def _(
    ID_TO_DOMAIN,
    ID_TO_GENERATOR,
    OUTPUT_DIR,
    REBUTTAL_DIR,
    SHARED_BACKBONES,
    discrimination_metrics,
    make_cluster_sampler,
    mo,
    np,
    os,
    pd,
    rebuttal_df,
    rebuttal_tp_df,
):
    def _stratum_metrics(group, n_resamples=1000, seed=42):
        y = (group['human_judgment'] == 'Incomplete').values
        p = (group['model_judgment'] == 'Incomplete').values
        s = group['score'].values
        point = discrimination_metrics(y, p, s)
        draw = make_cluster_sampler(group['qid'].values)
        rng = np.random.default_rng(seed)
        boots = pd.DataFrame([discrimination_metrics(y[i], p[i], s[i]) for i in (draw(rng) for _ in range(n_resamples))])
        out = {'n': len(group), 'prevalence_incomplete': y.mean(), **point}
        for stat in ('F1', 'AUC-PR', 'AUROC'):
            out[f'{stat}_lo'], out[f'{stat}_hi'] = np.nanpercentile(boots[stat].values, [2.5, 97.5])
        return pd.Series(out)

    _breakdowns = {}
    for _col, _id_map in (('domain', ID_TO_DOMAIN), ('generator', ID_TO_GENERATOR)):
        _verdict = (
            rebuttal_df.groupby(['metric', 'eval_model', _col])
            .apply(_stratum_metrics, include_groups=False)
            .reset_index()
        )
        _tp = rebuttal_tp_df.assign(**{_col: rebuttal_tp_df['ID'].map(_id_map)})
        _tp = _tp[_tp['Eval Model'].isin(SHARED_BACKBONES)]
        _align = (
            _tp.groupby(['Metric', _col])['aligned']
            .agg(n_tp_shared3='size', pct_yes_shared3=lambda a: 100 * (a == 'Yes').mean())
            .reset_index()
            .rename(columns={'Metric': 'metric'})
        )
        _out = _verdict.merge(_align, on=['metric', _col], how='left')
        _out.to_csv(os.path.join(REBUTTAL_DIR, f'breakdown_{_col}.csv'), index=False)
        _breakdowns[_col] = _out

    # LaTeX for the domain and generator splits (main-text backbones only).
    _MAIN_JUDGES = SHARED_BACKBONES + ['Prometheus-2 7B']
    _DEFS = ['Omission-Harm Rubric', 'Criteria-Checklist']

    def _judge_rows(df):
        """(definition, judge, group) in paper order, skipping configurations that were not run."""
        for _metric in _DEFS:
            _judges = [j for j in _MAIN_JUDGES if ((df['metric'] == _metric) & (df['eval_model'] == j)).any()]
            for _i, _judge in enumerate(_judges):
                _label = r'\multirow{' + str(len(_judges)) + r'}{*}{' + _metric + r'}' if _i == 0 else ''
                yield _metric, _label, _judge, df[(df['metric'] == _metric) & (df['eval_model'] == _judge)]

    _d = _breakdowns['domain']
    _prev = _d[_d['eval_model'] == 'GPT-5 Mini'].groupby('domain')[['n', 'prevalence_incomplete']].first()
    _DOM_STATS = ('F1', 'Recall', 'Precision', 'AUC-PR')
    _tex = [
        r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{ll' + 'cc' * len(_DOM_STATS) + r'}', r'    \toprule',
        r'    & & ' + ' & '.join(r'\multicolumn{2}{c}{\textbf{' + _st + r'}}' for _st in _DOM_STATS) + r' \\',
        r'    ' + ' '.join(f'\\cmidrule(lr){{{3 + 2 * _k}-{4 + 2 * _k}}}' for _k in range(len(_DOM_STATS))),
        r'    \textbf{Definition} & \textbf{Judge} & ' + ' & '.join(['MH & PC'] * len(_DOM_STATS)) + r' \\', r'    \midrule',
    ]
    for _mi, _metric in enumerate(_DEFS):
        _rows_m = list(_judge_rows(_d[_d['metric'] == _metric]))
        if not _rows_m:
            continue
        if _mi > 0:
            _tex.append(r'    \midrule')
        _n_rows = len(_rows_m) + 1
        for _i, (_, _, _judge, _g) in enumerate(_rows_m):
            _label = r'\multirow{' + str(_n_rows) + r'}{*}{' + _metric + r'}' if _i == 0 else ''
            _by_dom = _g.set_index('domain')
            _cells = [f"{100 * _by_dom.loc[_dom, _st]:.1f}" for _st in _DOM_STATS for _dom in ('MH', 'PC')]
            _tex.append(f"    {_label} & {_judge} & " + ' & '.join(_cells) + r' \\')
        # All: mean over the backbones above, per domain.
        _sub = _d[(_d['metric'] == _metric) & _d['eval_model'].isin([j for _, _, j, _ in _rows_m])]
        _mean = _sub.groupby('domain')[list(_DOM_STATS)].mean()
        _tex.append(r'    \cmidrule(l){2-' + str(2 + 2 * len(_DOM_STATS)) + r'}')
        _tex.append(r'     & \textit{All} & ' + ' & '.join(f"{100 * _mean.loc[_dom, _st]:.1f}" for _st in _DOM_STATS for _dom in ('MH', 'PC')) + r' \\')
    _tex += [
        r'    \bottomrule', r'\end{tabular}',
        r'\caption{Verdict-level performance by MedExpert domain, i.e., the medical specialty of the patient question '
        r'(Incomplete class, scaled by 100). MH: mental health, $n = '
        + str(int(_prev.loc['MH', 'n'])) + r'$, ' + f"{100 * _prev.loc['MH', 'prevalence_incomplete']:.0f}" + r'\% '
        r'incomplete; PC: prenatal care, $n = ' + str(int(_prev.loc['PC', 'n'])) + r'$, '
        + f"{100 * _prev.loc['PC', 'prevalence_incomplete']:.0f}" + r'\% incomplete. \textit{All}: mean over the '
        r'backbones above. Precision and AUC-PR stay near the domain\textquotesingle s incomplete rate (the random '
        r'AUC-PR baseline) in both domains.}\label{tab:breakdown_domain}',
        r'\end{table}',
    ]
    # Written to OUTPUT_DIR (not REBUTTAL_DIR) so the latex-build copy step syncs it to the paper.
    with open(os.path.join(OUTPUT_DIR, 'breakdown_domain.tex'), 'w') as _f:
        _f.write('\n'.join(_tex) + '\n')

    # Generator split: F1 (Incomplete class) per response generator, the paper's primary verdict metric.
    _GEN_NAMES = {'llama2': 'Llama 2', 'llama3': 'Llama 3', 'gemma2': 'Gemma 2', 'olmo2': 'OLMo 2', 'biollm': 'OpenBioLLM'}
    _gdf = _breakdowns['generator']
    _gens = [g for g in _GEN_NAMES if g in set(_gdf['generator'])]
    _gprev = _gdf[_gdf['eval_model'] == 'GPT-5 Mini'].groupby('generator')[['n', 'prevalence_incomplete']].first()
    _tex = [
        r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{ll' + 'c' * len(_gens) + r'}', r'    \toprule',
        r'    & & \multicolumn{' + str(len(_gens)) + r'}{c}{\textbf{F1 by response generator}} \\',
        r'    \cmidrule(lr){3-' + str(2 + len(_gens)) + r'}',
        r'    \textbf{Definition} & \textbf{Judge} & ' + ' & '.join(
            r'\shortstack{' + _GEN_NAMES[g] + r' \\ {\scriptsize ' + f"{100 * _gprev.loc[g, 'prevalence_incomplete']:.0f}" + r'\%}}'
            for g in _gens
        ) + r' \\',
        r'    \midrule',
    ]
    _prev_metric = None
    for _metric, _label, _judge, _g in _judge_rows(_gdf):
        if _prev_metric is not None and _metric != _prev_metric:
            _tex.append(r'    \midrule')
        _prev_metric = _metric
        _by_gen = _g.set_index('generator')
        _tex.append(f"    {_label} & {_judge} & " + ' & '.join(f"{100 * _by_gen.loc[g, 'F1']:.1f}" for g in _gens) + r' \\')
    _max_f1 = 100 * _gdf[_gdf['eval_model'].isin(_MAIN_JUDGES)]['F1'].max()
    _tex += [
        r'    \bottomrule', r'\end{tabular}',
        r'\caption{F1 for the Incomplete class (scaled by 100) by the chatbot that generated the MedExpert response; the percentage under '
        r'each generator is the share of its responses that the clinician rated incomplete. Each generator answered every question '
        r'($n \approx ' + str(int(round(_gprev['n'].mean()))) + r'$ per cell). No stratum exceeds '
        + f'{_max_f1:.1f}' + r'.}\label{tab:breakdown_generator}',
        r'\end{table}',
    ]
    with open(os.path.join(REBUTTAL_DIR, 'breakdown_generator.tex'), 'w') as _f:
        _f.write('\n'.join(_tex) + '\n')

    mo.vstack([
        mo.md('**By domain**'),
        mo.ui.table(_breakdowns['domain'].round(3), selection=None, page_size=len(_breakdowns['domain'])),
        mo.md('**By response generator**'),
        mo.ui.table(_breakdowns['generator'].round(3), selection=None, page_size=len(_breakdowns['generator'])),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §5 — Clinician–clinician vs. judge–clinician agreement (32 double-annotated responses)

    MedExpert (`sonal-ssj/MedExpert`, config `medexpert-all`) double-annotated 32 responses.
    Clinician A is the label used in the paper; clinician B is the second annotation.
    Produced by `tools/extract_medexpert_double_annotations.py`. Binary labels use the paper's
    relaxed rule (no omissions or only Mild = Complete). CIs: percentile bootstrap over the
    32 items (2,000 resamples).
    """)
    return


@app.cell
def _(REBUTTAL_DIR, kappa_alpha_binary, krippendorff, mo, np, os, pd, rebuttal_df):
    _path = os.path.join(REBUTTAL_DIR, 'medexpert_double_annotations.jsonl')
    if not os.path.exists(_path):
        clinician_pair_df = pd.DataFrame()
        _out = mo.callout(mo.md(f'`{_path}` not found — run `tools/extract_medexpert_double_annotations.py`.'), kind='warn')
    else:
        _pairs = pd.read_json(_path, lines=True)
        # Keep responses in the analysis set (the loader drops the 10 few-shot example IDs).
        _pairs = _pairs[_pairs['id'].isin(set(rebuttal_df['id']))].reset_index(drop=True)
        _human = rebuttal_df.drop_duplicates('id').set_index('id')['human_judgment']
        _mismatch = (_pairs['clinician_a_judgment'].values != _human.loc[_pairs['id']].values).sum()
        assert _mismatch == 0, f'{_mismatch} pairs where clinician A differs from the paper label'
        _a = (_pairs['clinician_a_judgment'] == 'Incomplete').values
        _b = (_pairs['clinician_b_judgment'] == 'Incomplete').values

        def _agreement(r1, r2, n_resamples=2000, seed=42):
            rng = np.random.default_rng(seed)
            n = len(r1)
            point = ((r1 == r2).mean(), *kappa_alpha_binary(r1, r2))
            boots = np.array([
                ((r1[i] == r2[i]).mean(), *kappa_alpha_binary(r1[i], r2[i]))
                for i in (rng.integers(0, n, n) for _ in range(n_resamples))
            ])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
            return {
                'n': n, 'agreement': point[0], 'agreement_lo': lo[0], 'agreement_hi': hi[0],
                'kappa': point[1], 'kappa_lo': lo[1], 'kappa_hi': hi[1],
                'alpha': point[2], 'alpha_lo': lo[2], 'alpha_hi': hi[2],
                'pct_incomplete_r1': 100 * r1.mean(), 'pct_incomplete_r2': 100 * r2.mean(),
            }

        _rows = [{'comparison': 'Clinician A vs. Clinician B', 'metric': '', 'eval_model': '', **_agreement(_a, _b)}]
        for (_metric, _judge), _g in rebuttal_df[rebuttal_df['id'].isin(_pairs['id'])].groupby(['metric', 'eval_model']):
            _j = (_g.set_index('id').loc[_pairs['id'], 'model_judgment'] == 'Incomplete').values
            _rows.append({'comparison': 'Judge vs. Clinician A', 'metric': _metric, 'eval_model': _judge, **_agreement(_j, _a)})
            _rows.append({'comparison': 'Judge vs. Clinician B', 'metric': _metric, 'eval_model': _judge, **_agreement(_j, _b)})
        clinician_pair_df = pd.DataFrame(_rows)
        # Ordinal alpha on the five-level severity scale, clinician A vs. B.
        _ordinal_alpha = krippendorff.alpha(
            np.vstack([_pairs['clinician_a_score'].values, _pairs['clinician_b_score'].values]),
            level_of_measurement='ordinal',
        )
        clinician_pair_df.to_csv(os.path.join(REBUTTAL_DIR, 'clinician_vs_judge_agreement.csv'), index=False)

        # LaTeX: clinician pair, then every main-text Judge against each clinician.
        _f2 = lambda v: f'{v:.2f}'.replace('-0.00', '0.00')
        _ci = lambda r, k: f"{_f2(r[k])} [{_f2(r[k + '_lo'])}, {_f2(r[k + '_hi'])}]"
        _cc = clinician_pair_df.iloc[0]
        _tex = [
            r'\begin{table*}[t]', r'\centering', r'\small',
            r'\begin{tabular}{llcccc}', r'    \toprule',
            r'    & & \multicolumn{2}{c}{\textbf{vs.\ Clinician A (reference)}} & \multicolumn{2}{c}{\textbf{vs.\ Clinician B}} \\',
            r'    \cmidrule(lr){3-4} \cmidrule(lr){5-6}',
            r'    \textbf{Definition} & \textbf{Judge} & \textbf{Agreement} & $\boldsymbol{\alpha}$ \textbf{[95\% CI]} & \textbf{Agreement} & $\boldsymbol{\alpha}$ \textbf{[95\% CI]} \\',
            r'    \midrule',
            f"    \\multicolumn{{2}}{{l}}{{\\textit{{Clinician B}}}} & {100 * _cc['agreement']:.1f}\\% & {_ci(_cc, 'alpha')} & -- & -- \\\\",
            r'    \midrule',
        ]
        _judges = clinician_pair_df[(clinician_pair_df['comparison'] != 'Clinician A vs. Clinician B') & (clinician_pair_df['eval_model'] != 'OpenBioLLM')]
        for _mi, _metric in enumerate(['Omission-Harm Rubric', 'Criteria-Checklist']):
            _sub = _judges[_judges['metric'] == _metric]
            _models = list(dict.fromkeys(_sub['eval_model']))
            for _i, _judge in enumerate(_models):
                _ra = _sub[(_sub['eval_model'] == _judge) & (_sub['comparison'] == 'Judge vs. Clinician A')].iloc[0]
                _rb = _sub[(_sub['eval_model'] == _judge) & (_sub['comparison'] == 'Judge vs. Clinician B')].iloc[0]
                _label = r'\multirow{' + str(len(_models)) + r'}{*}{' + _metric + r'}' if _i == 0 else ''
                _tex.append(
                    f"    {_label} & {_judge} & {100 * _ra['agreement']:.1f}\\% & {_ci(_ra, 'alpha')} & "
                    f"{100 * _rb['agreement']:.1f}\\% & {_ci(_rb, 'alpha')} \\\\"
                )
            if _mi == 0:
                _tex.append(r'    \midrule')
        _tex += [
            r'    \bottomrule', r'\end{tabular}',
            r'\caption{Verdict agreement on the ' + str(int(_cc['n'])) + r' evaluation responses that two MedExpert '
            r'clinicians annotated independently (MedExpert \textit{All} release). Clinician A provides the labels used '
            r'throughout the paper. Binary verdicts use the paper\textquotesingle s rule (no omissions or only Mild = '
            r'Complete). $\alpha$: Krippendorff\textquotesingle s $\alpha$ (nominal), with a 95\% percentile bootstrap CI over the '
            r'responses (2{,}000 resamples). Clinicians rated ' + f"{_cc['pct_incomplete_r1']:.0f}" + r'\% (A) and '
            + f"{_cc['pct_incomplete_r2']:.0f}" + r'\% (B) of these responses incomplete.}\label{tab:clinician_vs_judge}',
            r'\end{table*}',
        ]
        with open(os.path.join(REBUTTAL_DIR, 'clinician_vs_judge_agreement.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')
        _out = mo.vstack([
            mo.md(
                f'**{len(_pairs)} double-annotated responses** (generator: {", ".join(sorted(_pairs["model"].unique()))}). '
                f'Clinician A vs. B ordinal α on severity: **{_ordinal_alpha:.3f}**. '
                f'Severity cross-tab:'
            ),
            mo.ui.table(pd.crosstab(_pairs['clinician_a_severity'], _pairs['clinician_b_severity']).reset_index(), selection=None),
            mo.ui.table(clinician_pair_df.round(3), selection=None, page_size=len(clinician_pair_df)),
        ])
    _out
    return (clinician_pair_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §4 — Taxonomy-validation "upgrades" on the MedExpert TP rows

    Cross-tab of classifier vs. human alignment labels on the 16 MedExpert TP validation rows
    used for the α table (the paper's upgrade sentence counted the 30-row TP/Agrees file,
    which includes HealthBench rows).
    """)
    return


@app.cell
def _(OUTPUT_DIR, REBUTTAL_DIR, load_validation_rows, mo, os, pd):
    _tp_path = os.path.join(
        OUTPUT_DIR, 'qualitative_analysis', 'annotated', 'rq2_qualitative_validation_tp_agrees_resampled_annotated.jsonl'
    )
    _tp = pd.DataFrame([r for r in load_validation_rows(_tp_path) if r['analysis_category'] == 'TP'])
    _order = ['Yes', 'Partially', 'No']
    validation_upgrade_df = (
        pd.crosstab(_tp['assigned_class'].rename('classifier'), _tp['human_omissions_aligned'].rename('human'))
        .reindex(index=_order, columns=_order, fill_value=0)
    )
    validation_upgrade_df.to_csv(os.path.join(REBUTTAL_DIR, 'validation_tp_upgrades_medexpert.csv'))
    mo.vstack([
        mo.md(f'**{len(_tp)} MedExpert TP validation rows** (rows: classifier label; columns: human label)'),
        mo.ui.table(validation_upgrade_df.reset_index(), selection=None),
    ])
    return (validation_upgrade_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §6 — Existing runs without the "actively search" directive (no new runs)

    The directive appears only in the few-shot Omission-Harm prompt. The zero-shot
    Omission-Harm prompt (no directive, no examples, no reasoning scaffold) and the
    Criteria-Checklist prompt (no directive) were already run on the same responses. This is a
    confounded comparison, not an ablation: each prompt differs in more than the directive.
    """)
    return


@app.cell
def _(
    REBUTTAL_BACKBONES,
    REBUTTAL_DIR,
    RQ2_ANALYSIS_RESULTS_PATH,
    completeness_accuracy_df,
    discrimination_metrics,
    mo,
    os,
    pd,
):
    _METRICS = ['Omission-Harm Rubric', 'Omission-Harm Rubric (Zero-shot)', 'Criteria-Checklist']
    _sub = completeness_accuracy_df[
        (completeness_accuracy_df['dataset'] == 'MedExpert')
        & completeness_accuracy_df['metric'].isin(_METRICS)
        & completeness_accuracy_df['eval_model'].isin(REBUTTAL_BACKBONES)
    ]
    _rows = []
    for (_metric, _judge), _g in _sub.groupby(['metric', 'eval_model']):
        _y = (_g['human_judgment'] == 'Incomplete').values
        _p = (_g['model_judgment'] == 'Incomplete').values
        _rows.append({
            'metric': _metric, 'eval_model': _judge, 'n': len(_g),
            'directive_in_prompt': _metric == 'Omission-Harm Rubric',
            'pct_flagged_incomplete': 100 * _p.mean(),
            **discrimination_metrics(_y, _p, _g['score'].values),
        })
    directive_comparison_df = pd.DataFrame(_rows)
    directive_comparison_df.to_csv(os.path.join(REBUTTAL_DIR, 'directive_existing_runs.csv'), index=False)

    # LaTeX: main-text backbones only.
    _PROMPT_LABEL = {
        'Omission-Harm Rubric': r'\shortstack[l]{Omission-Harm \\ (few-shot)}',
        'Omission-Harm Rubric (Zero-shot)': r'\shortstack[l]{Omission-Harm \\ (zero-shot)}',
        'Criteria-Checklist': 'Criteria-Checklist',
    }
    _JUDGES = ['LLaMA 3.3-70B', 'GPT-5 Mini', 'Qwen3.6', 'Prometheus-2 7B']
    _sub_all = _sub
    _tex = [
        r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lclccc}', r'    \toprule',
        r'    \textbf{Prompt} & \textbf{Directive} & \textbf{Judge} & \textbf{\% flagged} & \textbf{AUC-PR} & \textbf{AUROC} \\',
        r'    \midrule',
    ]
    for _pi, _metric in enumerate(_METRICS):
        _sub = directive_comparison_df[(directive_comparison_df['metric'] == _metric)].set_index('eval_model')
        _judges = [j for j in _JUDGES if j in _sub.index]
        for _i, _judge in enumerate(_judges):
            _r = _sub.loc[_judge]
            _lead = (
                r'\multirow{' + str(len(_judges)) + r'}{*}{' + _PROMPT_LABEL[_metric] + r'} & \multirow{'
                + str(len(_judges)) + r'}{*}{' + (r'\checkmark' if _r['directive_in_prompt'] else r'$\times$') + '}'
            ) if _i == 0 else ' & '
            _tex.append(
                f"    {_lead} & {_judge} & {_r['pct_flagged_incomplete']:.1f} & {100 * _r['AUC-PR']:.1f} & {100 * _r['AUROC']:.1f} \\\\"
            )
        if _pi < len(_METRICS) - 1:
            _tex.append(r'    \midrule')
    _clinician_prevalence = 100 * (
        _sub_all.drop_duplicates('id')['human_judgment'] == 'Incomplete'
    ).mean()
    _tex += [
        r'    \bottomrule', r'\end{tabular}',
        r"\caption{Existing runs with and without the ``actively search for missing information'' directive "
        r'(no new runs). \% flagged: share of responses the Judge rated incomplete (clinicians: '
        + f'{_clinician_prevalence:.0f}' + r'\%); AUC-PR and AUROC are scaled by 100 (random: '
        + f'{_clinician_prevalence:.0f}' + r' and 50). The prompts also differ in examples, reasoning scaffold and, for '
        r'the Criteria-Checklist, Judge-generated criteria, so this is a confounded comparison rather than an '
        r'ablation.}\label{tab:directive_existing_runs}',
        r'\end{table}',
    ]
    with open(os.path.join(REBUTTAL_DIR, 'directive_existing_runs.tex'), 'w') as _f:
        _f.write('\n'.join(_tex) + '\n')

    # Classifier failure-mode codes on FP rows (clinician Complete, judge Incomplete), per prompt.
    # Read from the classifier output directly: the enriched frame omits the zero-shot rows.
    _res = pd.read_json(RQ2_ANALYSIS_RESULTS_PATH, lines=True)
    _res['Metric'] = _res['Metric'].replace({'Dynamic-Checklist': 'Criteria-Checklist'})
    _fp = _res[
        (_res['Dataset'] == 'MedExpert')
        & _res['Metric'].isin(_METRICS)
        & _res['Eval Model'].isin(REBUTTAL_BACKBONES)
        & _res['Human Judgment'].str.startswith('Complete')
        & _res['Model Judgment'].str.startswith('Incomplete')
    ].copy()
    _fp['code'] = _fp['model_response_parsed'].apply(lambda x: x.get('category') if isinstance(x, dict) else None)
    _fp['code'] = _fp['code'].replace({'FP05': 'Other'})
    directive_fp_codes_df = pd.crosstab(_fp['Metric'], _fp['code'], normalize='index').mul(100)
    directive_fp_codes_df['n_fp'] = _fp.groupby('Metric').size()
    directive_fp_codes_df.to_csv(os.path.join(REBUTTAL_DIR, 'directive_existing_runs_fp_codes.csv'))
    mo.vstack([
        mo.ui.table(directive_comparison_df.round(3), selection=None, page_size=len(directive_comparison_df)),
        mo.md('**FP failure-mode codes (% of FP rows) by prompt, all five backbones**'),
        mo.ui.table(directive_fp_codes_df.round(1).reset_index(), selection=None),
    ])
    return directive_comparison_df, directive_fp_codes_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example Judge-generated criteria (Criteria-Checklist)

    The criteria each shared backbone generated for two MedExpert responses, with the per-criterion
    verdict: the prenatal strength-training response in Figure 1 and one mental-health response.
    """)
    return


@app.cell
def _(REBUTTAL_DIR, SHARED_BACKBONES, mo, os, pd, re, results_df_all):
    # (title, short description) of the two example responses. PC-llama3-q16 is the Figure 1
    # example; MH-llama3-q54 has a Moderate clinician omission about alcohol-medication interactions.
    _EXAMPLES = [('PC-llama3-q16', 'prenatal care'), ('MH-llama3-q54', 'mental health')]
    _MAX_CHARS = 120

    def _tex_escape(text):
        text = str(text).replace('\\', r'\textbackslash{}')
        text = re.sub(r'([&%$#_{}])', r'\\\1', text)
        for _old, _new in (('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'), ('≤', r'$\leq$'),
                           ('≥', r'$\geq$'), ('…', r'\ldots{}'), ('µ', r'$\mu$'), ('≈', r'$\approx$'), ('→', r'$\rightarrow$'),
                           ('\u2011', '-'), ('\u2010', '-'), ('\u00a0', ' '), ('\u2009', ' '), ('\u202f', ' '),
                           ('–', '--'), ('—', '---'), ('×', r'$\times$'), ('±', r'$\pm$')):
            text = text.replace(_old, _new)
        return text

    def _shorten(text):
        text = ' '.join(str(text).split())
        return text if len(text) <= _MAX_CHARS else text[: _MAX_CHARS].rsplit(' ', 1)[0] + '…'

    _cc = results_df_all[
        (results_df_all['dataset'] == 'MedExpert') & (results_df_all['metric_display'] == 'Criteria-Checklist')
    ].copy()
    _cc['title'] = _cc['meta'].apply(lambda m: m.get('title') if isinstance(m, dict) else None)

    _tex = []
    _summary = []
    for _title, _domain in _EXAMPLES:
        _rows = _cc[_cc['title'] == _title]
        if _rows.empty:
            raise ValueError(f'no Criteria-Checklist rows for {_title}')
        _first = _rows.iloc[0]
        _omissions = '; '.join(_shorten(o) for o in (_first['omissions'] or [])) or 'none'
        _severity = str(_first['meta'].get('omission_severity') or 'No omissions').split(' - ')[0]
        _cols, _heads = [], []
        for _judge in SHARED_BACKBONES:
            _r = _rows[_rows['eval_model'] == _judge]
            if _r.empty:
                continue
            _r = _r.iloc[0]
            _crit = _r['metric_details'].get('meta') or []
            _met = sum(bool(c.get('criteria_score')) for c in _crit)
            _verdict = _r['model_judgment']
            _heads.append(
                r'\textbf{' + _judge + r'} \newline {\scriptsize ' + f'{_met}/{len(_crit)} met $\\rightarrow$ {_verdict}' + '}'
            )
            _cols.append(' '.join(
                (r'$\checkmark$ ' if c.get('criteria_score') else r'$\times$ ') + _tex_escape(_shorten(c.get('criterion', ''))) + r'\par'
                for c in _crit
            ))
            _summary.append({'title': _title, 'judge': _judge, 'n_criteria': len(_crit), 'n_met': _met, 'verdict': _verdict})
        _width = f'{0.90 / len(_cols):.3f}'
        _spec = ''.join(r'>{\raggedright\arraybackslash}p{' + _width + r'\linewidth}' for _ in _cols)
        _tex += [
            r'\begin{table*}[p]', r'\centering', r'\scriptsize',
            r'\begin{tabular}{' + _spec + r'}', r'    \toprule',
            '    ' + ' & '.join(_heads) + r' \\', r'    \midrule',
            '    ' + ' & '.join(_cols) + r' \\', r'    \bottomrule', r'\end{tabular}',
            r'\caption{Criteria generated and scored by each backbone under the Criteria-Checklist for one MedExpert '
            + _domain + r' response (\texttt{' + _tex_escape(_title) + r'}). Question: ``' + _tex_escape(_first['question'].strip())
            + r"''. Clinician (worst omission: " + _severity + r'): ' + _tex_escape(_omissions).rstrip('. ') + '.'
            + r' $\checkmark$/$\times$: criterion met/not met. Long criteria are truncated (\ldots).}'
            + r'\label{tab:cc_criteria_' + _title.split('-')[0].lower() + r'}',
            r'\end{table*}', '',
        ]
    with open(os.path.join(REBUTTAL_DIR, 'cc_generated_criteria_examples.tex'), 'w') as _f:
        _f.write('\n'.join(_tex) + '\n')
    _summary_df = pd.DataFrame(_summary)
    mo.ui.table(_summary_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §2, §3, §5 — Blinded validation sample (`tools/validate_rq2.html`, blind mode)

    One annotator, blinded to the classifier label, its explanation, the definition name and
    the backbone. Two item types:

    - **`tp_alignment`** (120 items): 20 matched pairs per shared backbone, simple random
      within backbone from the pairs that are TP under both definitions; each pair contributes
      its Omission-Harm and its Criteria-Checklist item. Human labels therefore estimate the
      matched-pair comparison directly.
    - **`fp_adjudication`** (70 items): 10 per (definition × backbone) stratum of
      over-penalization cases (clinician Complete, judge Incomplete); the annotator rates
      whether the judge's main flagged omission is clinically warranted.
    - **Test–retest**: 20 `tp_alignment` items from the first part of the file are repeated
      under new IDs in the second half.

    Responses already in any earlier validation file are excluded. The file is written once;
    delete it to regenerate.
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    REBUTTAL_BACKBONES,
    REBUTTAL_CC,
    REBUTTAL_OH,
    SHARED_BACKBONES,
    alignment_matched_df,
    full_enriched_df,
    json,
    mo,
    os,
    pd,
    random,
    re,
):
    _BLIND_DIR = os.path.join(OUTPUT_DIR, 'qualitative_analysis', 'ml4h_blind')
    BLIND_ANNOTATOR_PATH = os.path.join(_BLIND_DIR, 'rq2_blind_validation_annotator1.jsonl')
    BLIND_KEY_PATH = os.path.join(_BLIND_DIR, 'rq2_blind_validation_key.jsonl')
    _N_PAIRS_PER_BACKBONE = 20
    _N_FP_PER_STRATUM = 10
    _N_RETEST = 20
    _SEED = 42

    # Responses already shown to an annotator in any earlier validation round.
    _annotated_dir = os.path.join(OUTPUT_DIR, 'qualitative_analysis', 'annotated')
    _seen_ids = set()
    for _fname in sorted(os.listdir(_annotated_dir)):
        if not _fname.endswith('.jsonl'):
            continue
        with open(os.path.join(_annotated_dir, _fname)) as _f:
            for _line in _f:
                if not _line.strip():
                    continue
                _r = json.loads(_line)
                if any(str(_r.get(_k) or '').strip() for _k in ('human_omissions_aligned', 'human_category', 'human_dis_category')):
                    _seen_ids.add(_r.get('ID'))

    _enriched = full_enriched_df[full_enriched_df['Dataset'] == 'MedExpert'].set_index(['ID', 'Metric', 'Eval Model'])

    def _verdict_word(judgment):
        return str(judgment).split(' ')[0]

    def _strip_score_line(explanation):
        # Omission-Harm explanations open with "Score: N"; the verdict is shown separately.
        return re.sub(r'^\s*Score:\s*\d+\s*\n+', '', str(explanation))

    def _make_item(key, item_type):
        row = _enriched.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        parsed = row['model_response_parsed'] if isinstance(row['model_response_parsed'], dict) else {}
        blind = {
            'item_type': item_type,
            'Question': row['Question'],
            'Answer': row['Answer'],
            'clinician_verdict': _verdict_word(row['Human Judgment']),
            'clinician_explanation': row['Explanation of Human Judgment'],
            'judge_verdict': _verdict_word(row['Model Judgment']),
            'judge_explanation': _strip_score_line(row['Explanation of Model Judgment']),
        }
        hidden = {
            'item_type': item_type, 'ID': key[0], 'Metric': key[1], 'Eval Model': key[2],
            'classifier_label': parsed.get('omissions_aligned') if item_type == 'tp_alignment' else parsed.get('category'),
            'classifier_explanation': parsed.get('explanation'),
            'scoring_model': row.get('scoring_model'),
        }
        return blind, hidden

    _blind_exists = os.path.exists(BLIND_ANNOTATOR_PATH) or os.path.exists(
        BLIND_ANNOTATOR_PATH.replace('.jsonl', '_annotated.jsonl')
    )
    if _blind_exists:
        _out = mo.callout(
            mo.md(f'Blind sample already exists at `{BLIND_ANNOTATOR_PATH}` — not regenerated.'), kind='info'
        )
    else:
        # TP alignment: matched pairs on the shared backbones, simple random within backbone.
        _pairs = alignment_matched_df[
            alignment_matched_df['Eval Model'].isin(SHARED_BACKBONES) & ~alignment_matched_df['ID'].isin(_seen_ids)
        ]
        _sampled_pairs = (
            _pairs.groupby('Eval Model', group_keys=False)
            .apply(lambda g: g.sample(n=min(_N_PAIRS_PER_BACKBONE, len(g)), random_state=_SEED))
        )
        _items = []
        for _, _p in _sampled_pairs.iterrows():
            for _metric in (REBUTTAL_OH, REBUTTAL_CC):
                _blind, _hidden = _make_item((_p['ID'], _metric, _p['Eval Model']), 'tp_alignment')
                _hidden['pair_key'] = f"{_p['ID']}|{_p['Eval Model']}"
                _items.append((_blind, _hidden))

        # FP adjudication: over-penalization cases per (definition x backbone).
        _fp_pool = full_enriched_df[
            (full_enriched_df['Dataset'] == 'MedExpert')
            & full_enriched_df['Human Judgment'].str.startswith('Complete')
            & full_enriched_df['Model Judgment'].str.startswith('Incomplete')
            & ~full_enriched_df['ID'].isin(_seen_ids)
            & (
                ((full_enriched_df['Metric'] == REBUTTAL_OH) & full_enriched_df['Eval Model'].isin(REBUTTAL_BACKBONES[:4]))
                | ((full_enriched_df['Metric'] == REBUTTAL_CC) & full_enriched_df['Eval Model'].isin(SHARED_BACKBONES))
            )
        ]
        _fp_sampled = (
            _fp_pool.groupby(['Metric', 'Eval Model'], group_keys=False)
            .apply(lambda g: g.sample(n=min(_N_FP_PER_STRATUM, len(g)), random_state=_SEED))
        )
        for _, _r in _fp_sampled.iterrows():
            _items.append(_make_item((_r['ID'], _r['Metric'], _r['Eval Model']), 'fp_adjudication'))

        # Order: shuffle, then re-insert test-retest copies of first-part TP items into the
        # second half (originals come from the first 40% so every repeat is >= 10% later).
        _rng = random.Random(_SEED)
        _rng.shuffle(_items)
        _half = len(_items) // 2
        _early_tp = [i for i, (_b, _h) in enumerate(_items[: int(0.4 * len(_items))]) if _h['item_type'] == 'tp_alignment']
        _retest_src = sorted(_rng.sample(_early_tp, min(_N_RETEST, len(_early_tp))))
        _second = _items[_half:]
        _retest_pairs = []
        for _i in _retest_src:
            _copy = (dict(_items[_i][0]), {**_items[_i][1], 'is_retest': True})
            _second.insert(_rng.randint(0, len(_second)), _copy)
            _retest_pairs.append((_items[_i], _copy))
        _final = _items[:_half] + _second

        _blind_rows, _key_rows = [], []
        for _k, (_blind, _hidden) in enumerate(_final, start=1):
            _item_id = f'B{_k:03d}'
            _blind_rows.append({'item_id': _item_id, **_blind})
            _key_rows.append({'item_id': _item_id, 'is_retest': False, **_hidden})
        # Link each retest copy to its original by object identity in _final.
        _position = {id(_h): _k for _k, (_b, _h) in enumerate(_final)}
        for _orig, _copy in _retest_pairs:
            _key_rows[_position[id(_copy[1])]]['retest_of'] = _key_rows[_position[id(_orig[1])]]['item_id']
            _key_rows[_position[id(_copy[1])]]['is_retest'] = True

        _forbidden = {'assigned_class', 'omissions_aligned', 'category', 'llm_explanation', 'Metric', 'Eval Model', 'ID'}
        assert not any(_forbidden & set(_r) for _r in _blind_rows), 'blind file leaks hidden fields'
        os.makedirs(_BLIND_DIR, exist_ok=True)
        pd.DataFrame(_blind_rows).to_json(BLIND_ANNOTATOR_PATH, orient='records', lines=True, force_ascii=False)
        pd.DataFrame(_key_rows).to_json(BLIND_KEY_PATH, orient='records', lines=True, force_ascii=False)
        _summary = pd.DataFrame(_key_rows).groupby(['item_type', 'Metric', 'Eval Model', 'is_retest']).size().reset_index(name='n')
        _out = mo.vstack([
            mo.md(
                f'Wrote **{len(_blind_rows)} items** to `{BLIND_ANNOTATOR_PATH}` and the key to `{BLIND_KEY_PATH}` '
                f'({len(_seen_ids)} previously validated responses excluded).'
            ),
            mo.ui.table(_summary, selection=None, page_size=len(_summary)),
        ])
    _out
    return BLIND_ANNOTATOR_PATH, BLIND_KEY_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §2, §3, §5, §12 — Scoring the blinded validation

    Scores the reduced blind sample: 30 TP items (10 per classifier label) plus the 70 FP items,
    against `rq2_blind_validation_key_reduced.jsonl`. Because the TP items are stratified on the
    classifier label, agreement is reported per classifier label and reweighted to the pool mix.
    Set `ML4H_BLIND_ANNOTATED` / `ML4H_BLIND_KEY` to score a different export or key.
    """)
    return


@app.cell
def _(
    BLIND_ANNOTATOR_PATH,
    BLIND_KEY_PATH,
    ID_TO_QID,
    REBUTTAL_CC,
    REBUTTAL_DIR,
    REBUTTAL_OH,
    SHARED_BACKBONES,
    FAILURE_MODE_NAMES,
    FP_FAILURE_MODES,
    cohen_kappa_score,
    full_enriched_df,
    make_cluster_sampler,
    mo,
    np,
    os,
    pd,
    rebuttal_tp_df,
    simpledorff,
):
    # TP labels from annotator1, FP adjudication labels from the earlier full-sample pass.
    _annotated_path = os.environ.get(
        'ML4H_BLIND_ANNOTATED',
        BLIND_ANNOTATOR_PATH.replace('_annotator1.jsonl', '_annotator1-tp_opus-fp_annotated.jsonl'),
    )
    _key_path = os.environ.get('ML4H_BLIND_KEY', BLIND_KEY_PATH.replace('.jsonl', '_reduced.jsonl'))
    _ALIGN = ['Yes', 'Partially', 'No']

    def _agreement(df, a_col, b_col):
        """n, exact agreement, Cohen's kappa and Krippendorff's alpha (nominal) for two label columns."""
        d = df.dropna(subset=[a_col, b_col])
        if len(d) < 2:
            return {'n': len(d), 'agreement': np.nan, 'kappa': np.nan, 'alpha': np.nan}
        melted = pd.concat([
            pd.DataFrame({'item': d['item_id'], 'rater': 'a', 'label': d[a_col]}),
            pd.DataFrame({'item': d['item_id'], 'rater': 'b', 'label': d[b_col]}),
        ])
        try:
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(melted, experiment_col='item', annotator_col='rater', class_col='label')
        except ZeroDivisionError:
            alpha = np.nan
        return {
            'n': len(d),
            'agreement': float((d[a_col] == d[b_col]).mean()),
            'kappa': cohen_kappa_score(d[a_col], d[b_col]),
            'alpha': alpha,
        }

    if not os.path.exists(_annotated_path):
        blind_scores = {}
        _out = mo.callout(mo.md(f'No annotations yet at `{_annotated_path}`.'), kind='info')
    else:
        _ann = pd.read_json(_annotated_path, lines=True)
        _key = pd.read_json(_key_path, lines=True)
        _df = _key.merge(_ann[['item_id', 'human_label', 'human_notes', 'human_flagged']], on='item_id', how='left', validate='1:1')
        # TP items: the 40-item validation set (stage 1 = the original 30, sampled 10 per label of
        # the old classifier; stage 2 = 10 random items held out from prompt development), scored
        # against the final v5 classifier's labels from the full run, i.e. the labels the pool
        # rates below use (not the majority-of-3 validation runs in the iteration log).
        _tp40 = pd.read_json(BLIND_KEY_PATH.replace('.jsonl', '_tp40.jsonl'), lines=True).merge(
            pd.read_json(BLIND_ANNOTATOR_PATH.replace('.jsonl', '_tp40_annotated.jsonl'), lines=True)[
                ['item_id', 'human_label', 'human_notes', 'human_flagged']],
            on='item_id', how='left', validate='1:1',
        ).rename(columns={'classifier_label': 'classifier_label_v0'})
        _v5_label = rebuttal_tp_df.set_index(['ID', 'Metric', 'Eval Model'])['aligned']
        _tp40['classifier_label'] = [_v5_label.get(_k) for _k in zip(_tp40['ID'], _tp40['Metric'], _tp40['Eval Model'])]
        assert _tp40['classifier_label'].notna().all(), 'validation item without a v5 label'
        _df = pd.concat([_df[_df['item_type'] != 'tp_alignment'], _tp40], ignore_index=True)
        _df['human_label'] = _df['human_label'].where(_df['human_label'].notna() & (_df['human_label'] != ''))
        # FP adjudication labels can come from a separate export (the human relabel of the
        # 70 fp_adjudication items); TP labels always come from _annotated_path.
        _fp_labels_path = os.environ.get('ML4H_BLIND_FP_ANNOTATED')
        if _fp_labels_path:
            _fp_ann = pd.read_json(_fp_labels_path, lines=True)
            _fp_ann = _fp_ann[_fp_ann['human_label'].notna() & (_fp_ann['human_label'] != '')]
            _fp_map = dict(zip(_fp_ann['item_id'], _fp_ann['human_label']))
            _is_fp_item = _df['item_type'] == 'fp_adjudication'
            _df.loc[_is_fp_item, 'human_label'] = _df.loc[_is_fp_item, 'item_id'].map(_fp_map)
            print(f'FP adjudication labels from {_fp_labels_path}: {_df.loc[_is_fp_item, "human_label"].notna().sum()} labeled')
        _df['qid'] = _df['ID'].map(ID_TO_QID)
        _df['is_gpt_family'] = _df['Eval Model'] == 'GPT-5 Mini'
        _main = _df[~_df['is_retest']]
        _tp = _main[_main['item_type'] == 'tp_alignment']
        _fp = _main[_main['item_type'] == 'fp_adjudication']

        # (1) Classifier vs. human on TP alignment: overall, per definition, per backbone, GPT-family.
        _rows = [{'subset': 'All TP items', **_agreement(_tp, 'classifier_label', 'human_label')}]
        for _metric, _g in _tp.groupby('Metric'):
            _rows.append({'subset': _metric, **_agreement(_g, 'classifier_label', 'human_label')})
        for _judge, _g in _tp.groupby('Eval Model'):
            _rows.append({'subset': _judge, **_agreement(_g, 'classifier_label', 'human_label')})
        for _flag, _g in _tp.groupby('is_gpt_family'):
            _rows.append({'subset': 'GPT-5 Mini judge' if _flag else 'Non-GPT judges', **_agreement(_g, 'classifier_label', 'human_label')})
        # Ordered (Yes > Partially > No) weighting is a robustness check on the nominal alpha.
        _tp_ord = _tp.dropna(subset=['human_label', 'classifier_label'])
        _rows.append({
            'subset': 'All TP items (quadratic-weighted kappa)', 'n': len(_tp_ord),
            'kappa': cohen_kappa_score(
                _tp_ord['classifier_label'].map({'No': 0, 'Partially': 1, 'Yes': 2}),
                _tp_ord['human_label'].map({'No': 0, 'Partially': 1, 'Yes': 2}), weights='quadratic',
            ) if len(_tp_ord) else np.nan,
        })
        # The sample is stratified on the classifier label, so the per-label agreement
        # (precision of each classifier label) is the primary quantity; the pooled accuracy
        # reweights it to the classifier mix of the full TP pool (shared backbones).
        _pool_mix = rebuttal_tp_df[rebuttal_tp_df['Eval Model'].isin(SHARED_BACKBONES)]['aligned'].value_counts(normalize=True)
        _precision = _tp_ord.groupby('classifier_label').apply(lambda g: (g['human_label'] == g['classifier_label']).mean())
        for _label in _ALIGN:
            _g = _tp_ord[_tp_ord['classifier_label'] == _label]
            _rows.append({'subset': f'Classifier = {_label}', 'n': len(_g), 'agreement': _precision.get(_label, np.nan)})
        _rows.append({
            'subset': 'All TP items (reweighted to pool classifier mix)', 'n': len(_tp_ord),
            'agreement': sum(_pool_mix.get(c, 0) * _precision.get(c, np.nan) for c in _ALIGN),
        })
        _classifier_agreement_df = pd.DataFrame(_rows)

        # (2) Test-retest (intra-annotator) on repeated items.
        _retest = _df[_df['is_retest']].merge(
            _df[~_df['is_retest']][['item_id', 'human_label']].rename(columns={'item_id': 'retest_of', 'human_label': 'first_label'}),
            on='retest_of',
        )
        _retest_agreement = _agreement(_retest, 'first_label', 'human_label')

        # (3) Confusion matrix, classifier x human.
        _confusion_df = pd.crosstab(_tp['classifier_label'], _tp['human_label']).reindex(index=_ALIGN, columns=_ALIGN, fill_value=0)

        # (4) Human-label matched-pair comparison (shared backbones), question-clustered.
        _wide = _tp.pivot_table(index=['pair_key', 'qid', 'Eval Model'], columns='Metric', values='human_label', aggfunc='first').dropna().reset_index()
        # Skipped for the reduced sample, which keeps only a few complete pairs.
        _pair_rows = []
        for _label_set, _name in ((['Yes'], 'Yes'), (['Yes', 'Partially'], 'Yes or Partially')) if len(_wide) >= 10 else ():
            _x_oh = _wide[REBUTTAL_OH].isin(_label_set).values
            _x_cc = _wide[REBUTTAL_CC].isin(_label_set).values
            _draw = make_cluster_sampler(_wide['qid'].values)
            _rng = np.random.default_rng(42)
            _diffs = np.array([_x_cc[i].mean() - _x_oh[i].mean() for i in (_draw(_rng) for _ in range(10000))])
            _lo, _hi = np.percentile(_diffs, [2.5, 97.5])
            _pair_rows.append({
                'outcome': _name, 'n_pairs': len(_wide), 'human_rate_oh': 100 * _x_oh.mean(), 'human_rate_cc': 100 * _x_cc.mean(),
                'diff_cc_minus_oh': 100 * (_x_cc.mean() - _x_oh.mean()), 'diff_ci95_lo': 100 * _lo, 'diff_ci95_hi': 100 * _hi,
            })
        _human_matched_df = pd.DataFrame(_pair_rows)

        # (5) Misclassification-corrected rates for the full TP pools (shared backbones):
        # P(human = h | classifier = c) from the sample, weighted by the pool's classifier mix.
        _tp_lab = _tp.dropna(subset=['human_label']).reset_index(drop=True)
        _pools = {
            _metric: _pool[_pool['Eval Model'].isin(SHARED_BACKBONES)]
            for _metric, _pool in (
                (REBUTTAL_OH, rebuttal_tp_df[rebuttal_tp_df['Metric'] == REBUTTAL_OH]),
                (REBUTTAL_CC, rebuttal_tp_df[rebuttal_tp_df['Metric'] == REBUTTAL_CC]),
                ('Both', rebuttal_tp_df),
            )
        }
        _mixes = {_m: _p['aligned'].value_counts(normalize=True) for _m, _p in _pools.items()}

        def _corrected_stats(classifier, human):
            """Corrected Yes/No rates per pool and the reweighted agreement, from label arrays.

            P(human = h | classifier = c) is estimated on the validation sample and weighted
            by each pool's classifier-label mix (the pool mix is held fixed).
            """
            out = {}
            for _m, _mix in _mixes.items():
                for _h in ('Yes', 'No'):
                    out[(_m, _h)] = 100 * sum(
                        _mix.get(c, 0) * ((human[classifier == c] == _h).mean() if (classifier == c).any() else np.nan)
                        for c in _ALIGN
                    )
            out['reweighted_agreement'] = sum(
                _pool_mix.get(c, 0) * ((human[classifier == c] == c).mean() if (classifier == c).any() else np.nan)
                for c in _ALIGN
            )
            return out

        # Bootstrap CIs: resample items within each classifier label, so every draw keeps the
        # number of items per classifier label fixed.
        _cls = _tp_lab['classifier_label'].values
        _hum = _tp_lab['human_label'].values
        _strata = [np.flatnonzero(_cls == c) for c in _ALIGN if (_cls == c).any()]
        _point = _corrected_stats(_cls, _hum)
        _rng = np.random.default_rng(20261011)
        _boot = []
        for _ in range(10000):
            _idx = np.concatenate([_rng.choice(s, size=len(s), replace=True) for s in _strata])
            _boot.append(_corrected_stats(_cls[_idx], _hum[_idx]))
        _boot = pd.DataFrame(_boot)
        _ci = {k: np.nanpercentile(_boot[k].values, [2.5, 97.5]) for k in _point}

        _corr_rows = []
        for _metric, _pool in _pools.items():
            _mix = _mixes[_metric]
            _corr_rows.append({
                'definition': _metric, 'n_pool': len(_pool),
                'classifier_pct_yes': 100 * _mix.get('Yes', 0),
                'corrected_pct_yes': _point[(_metric, 'Yes')],
                'corrected_pct_yes_lo': _ci[(_metric, 'Yes')][0], 'corrected_pct_yes_hi': _ci[(_metric, 'Yes')][1],
                'classifier_pct_no': 100 * _mix.get('No', 0),
                'corrected_pct_no': _point[(_metric, 'No')],
                'corrected_pct_no_lo': _ci[(_metric, 'No')][0], 'corrected_pct_no_hi': _ci[(_metric, 'No')][1],
            })
        _corrected_rates_df = pd.DataFrame(_corr_rows)
        _reweighted = {
            'agreement': _point['reweighted_agreement'],
            'lo': _ci['reweighted_agreement'][0], 'hi': _ci['reweighted_agreement'][1],
        }
        _is_rw = _classifier_agreement_df['subset'] == 'All TP items (reweighted to pool classifier mix)'
        _classifier_agreement_df['agreement_lo'] = np.where(_is_rw, _reweighted['lo'], np.nan)
        _classifier_agreement_df['agreement_hi'] = np.where(_is_rw, _reweighted['hi'], np.nan)

        # (6) FP adjudication distribution and cross-tab with the classifier's FP codes.
        _fp_adjudication_df = pd.crosstab([_fp['Metric'], _fp['Eval Model']], _fp['human_label'], margins=True)
        _fp_vs_code_df = pd.crosstab(_fp['classifier_label'].replace({'FP05': 'Other'}), _fp['human_label'], margins=True)
        # Same 70 items against the final d1 label (rows: d1 label; columns: human category).
        _d1_label = full_enriched_df.set_index(['ID', 'Metric', 'Eval Model'])['failure_mode']
        _fp = _fp.assign(d1_label=[_d1_label.get(_k) for _k in zip(_fp['ID'], _fp['Metric'], _fp['Eval Model'])])
        assert _fp['d1_label'].notna().all(), 'FP adjudication item without a d1 label'
        _fp_vs_d1_df = pd.crosstab(_fp['d1_label'], _fp['human_label'], margins=True)
        _fp_vs_d1_df = _fp_vs_d1_df.reindex([c for c in FP_FAILURE_MODES if c in _fp_vs_d1_df.index] + ['All'])
        _fp_vs_d1_df.index = [FAILURE_MODE_NAMES.get(_i, _i) for _i in _fp_vs_d1_df.index]
        _fp_vs_d1_df.index.name = 'd1_label'

        # ── LaTeX outputs ─────────────────────────────────────────────────────
        _SHORT = {REBUTTAL_OH: 'Omission-Harm Rubric', REBUTTAL_CC: 'Criteria-Checklist'}
        _agr = _classifier_agreement_df.set_index('subset')
        _fmt2 = lambda v: '--' if pd.isna(v) else f'{v:.2f}'

        # (a) Classifier vs. blinded human agreement on the TP alignment items.
        _tex = [
            r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
            r'\begin{tabular}{lrccc}', r'    \toprule',
            r'    \textbf{TP items} & $\boldsymbol{n}$ & \textbf{Agreement} & $\boldsymbol{\kappa}$ & $\boldsymbol{\alpha}$ \\',
            r'    \midrule',
        ]
        _subsets = [('All TP items', 'All')] + [(m, _SHORT[m]) for m in (REBUTTAL_OH, REBUTTAL_CC)] + [(b, b + ' Judge') for b in SHARED_BACKBONES]
        for _i, (_key, _label) in enumerate(_subsets):
            _r = _agr.loc[_key]
            _tex.append(f"    {_label} & {int(_r['n'])} & {100 * _r['agreement']:.1f}\\% & {_fmt2(_r['kappa'])} & {_fmt2(_r['alpha'])} \\\\")
            if _i in (0, 2):
                _tex.append(r'    \midrule')
        _rw = _agr.loc['All TP items (reweighted to pool classifier mix)']
        _qwk = _agr.loc['All TP items (quadratic-weighted kappa)', 'kappa']
        _tex += [
            r'    \bottomrule', r'\end{tabular}',
            r'\caption{Agreement between the GPT-5.4~Mini alignment classifier and a blinded human annotator on '
            f"{int(_agr.loc['All TP items', 'n'])}"
            r' True Positive pairs from the three backbones run with both prompt configurations: 30 sampled '
            r'10 per label of an earlier classifier version, and 10 sampled at random and held out from prompt '
            r'development. The annotator saw neither '
            r'the classifier\textquotesingle s label nor its explanation, the prompt configuration or the backbone. '
            r'$\kappa$: Cohen\textquotesingle s $\kappa$; $\alpha$: Krippendorff\textquotesingle s $\alpha$ (nominal). '
            f'Quadratic-weighted $\\kappa$ (ordering \\textit{{No}} $<$ \\textit{{Partially}} $<$ \\textit{{Yes}}): {_qwk:.2f}. '
            r'Reweighted to the classifier-label mix of the full TP pool, agreement is '
            f"{100 * _rw['agreement']:.1f}\\% [{100 * _rw['agreement_lo']:.1f}, {100 * _rw['agreement_hi']:.1f}]"
            r' (95\% bootstrap CI, resampling within classifier label). Subgroups have '
            f"{min(int(_agr.loc[_k, 'n']) for _k, _ in _subsets[1:])}--{max(int(_agr.loc[_k, 'n']) for _k, _ in _subsets[1:])} items.}}"
            r'\label{tab:rq2_blind_validation}',
            r'\end{table}',
        ]
        with open(os.path.join(REBUTTAL_DIR, 'rq2_blind_validation.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')
        _tex_agreement = _tex

        # (b) Confusion matrix, classifier (rows) x human (columns).
        _cm = _confusion_df.copy()
        _cm['Total'] = _cm.sum(axis=1)
        _cm.loc['Total'] = _cm.sum(axis=0)
        _tex = [
            r'\begin{table}[t]', r'\centering', r'\small',
            r'\begin{tabular}{lcccc}', r'    \toprule',
            r'    & \multicolumn{3}{c}{\textbf{Human label}} & \\', r'    \cmidrule(lr){2-4}',
            r'    \textbf{Classifier label} & \textit{Yes} & \textit{Partially} & \textit{No} & \textbf{Total} \\',
            r'    \midrule',
        ]
        for _c in _ALIGN + ['Total']:
            if _c == 'Total':
                _tex.append(r'    \midrule')
            _vals = [
                (r'\textbf{' + str(int(_cm.loc[_c, _h])) + '}') if _c == _h and _c in _ALIGN else str(int(_cm.loc[_c, _h]))
                for _h in _ALIGN + ['Total']
            ]
            _tex.append(f"    {_c if _c == 'Total' else chr(92) + 'textit{' + _c + '}'} & " + ' & '.join(_vals) + r' \\')
        _n_up = int(sum(_confusion_df.loc[c, h] for i, c in enumerate(_ALIGN) for j, h in enumerate(_ALIGN) if j > i))
        _n_down = int(sum(_confusion_df.loc[c, h] for i, c in enumerate(_ALIGN) for j, h in enumerate(_ALIGN) if j < i))
        _tex += [
            r'    \bottomrule', r'\end{tabular}',
            r'\caption{Classifier vs.\ blinded human alignment labels on the TP validation items (bold: agreement). '
            f'The classifier assigned a higher alignment label than the annotator in {_n_up} items and a lower one in {_n_down}.'
            r'}\label{tab:rq2_blind_confusion}',
            r'\end{table}',
        ]
        with open(os.path.join(REBUTTAL_DIR, 'rq2_blind_confusion.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')
        _tex_confusion = _tex

        # (a+b) The paper shows (a) and (b) as two subtables of one table; the files above
        # are still written. Labels: the table is tab:rq2_blind_validation, the subtables
        # tab:rq2_blind_agreement and tab:rq2_blind_confusion.
        _tabular = lambda lines: lines[
            next(i for i, l in enumerate(lines) if l.startswith(r'\begin{tabular}')):lines.index(r'\end{tabular}') + 1
        ]
        _sub_n = [int(_agr.loc[_k, 'n']) for _k, _ in _subsets[1:]]
        _tex = [
            r'\begin{table}[t]', r'\centering', r'\small',
            r'\begin{subtable}[t]{0.53\linewidth}', r'\centering', r'\setlength{\tabcolsep}{4pt}',
            *_tabular(_tex_agreement),
            r'\caption{Agreement, overall and by subgroup.}\label{tab:rq2_blind_agreement}',
            r'\end{subtable}\hfill',
            r'\begin{subtable}[t]{0.44\linewidth}', r'\centering',
            *_tabular(_tex_confusion),
            r'\caption{Classifier (rows) vs.\ human (columns) labels; bold: agreement.}\label{tab:rq2_blind_confusion}',
            r'\end{subtable}',
            r'\caption{Blinded validation of the GPT-5.4~Mini reasoning-alignment classifier on '
            f"{int(_agr.loc['All TP items', 'n'])}"
            r' True Positive pairs from the three backbones run with both prompt configurations: 30 sampled '
            r'10 per label of an earlier classifier version, and 10 sampled at random and held out from prompt '
            r'development. The annotator saw neither '
            r'the classifier\textquotesingle s label nor its explanation, the prompt configuration or the backbone. '
            r'(a) $\kappa$: Cohen\textquotesingle s $\kappa$; $\alpha$: Krippendorff\textquotesingle s $\alpha$ (nominal). '
            f'Quadratic-weighted $\\kappa$ (ordering \\textit{{No}} $<$ \\textit{{Partially}} $<$ \\textit{{Yes}}): {_qwk:.2f}. '
            r'Reweighted to the classifier-label mix of the full TP pool, agreement is '
            f"{100 * _rw['agreement']:.1f}\\% [{100 * _rw['agreement_lo']:.1f}, {100 * _rw['agreement_hi']:.1f}]"
            r' (95\% bootstrap CI, resampling within classifier label). Subgroups have '
            f'{min(_sub_n)}--{max(_sub_n)} items. '
            f'(b) The classifier assigned a higher alignment label than the annotator in {_n_up} items and a lower one in {_n_down}.'
            r'}\label{tab:rq2_blind_validation}',
            r'\end{table}',
        ]
        with open(os.path.join(OUTPUT_DIR, 'rq2_blind_validation_combined.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')

        # (c) Classifier vs. misclassification-corrected alignment rates.
        _tex = [
            r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
            r'\begin{tabular}{lrcccc}', r'    \toprule',
            r'    & & \multicolumn{2}{c}{\textbf{Full alignment (\textit{Yes})}} & \multicolumn{2}{c}{\textbf{No shared reasoning (\textit{No})}} \\',
            r'    \cmidrule(lr){3-4} \cmidrule(lr){5-6}',
            r'    \textbf{Definition} & $\boldsymbol{n}$ & \textbf{Classifier} & \textbf{Corrected} & \textbf{Classifier} & \textbf{Corrected} \\',
            r'    \midrule',
        ]
        for _, _r in _corrected_rates_df.iterrows():
            _label = _SHORT.get(_r['definition'], r'\textit{Both}')
            _tex.append(
                f"    {_label} & {int(_r['n_pool'])} & {_r['classifier_pct_yes']:.1f} & "
                f"{_r['corrected_pct_yes']:.1f} [{_r['corrected_pct_yes_lo']:.1f}, {_r['corrected_pct_yes_hi']:.1f}] & "
                f"{_r['classifier_pct_no']:.1f} & "
                f"{_r['corrected_pct_no']:.1f} [{_r['corrected_pct_no_lo']:.1f}, {_r['corrected_pct_no_hi']:.1f}] \\\\"
            )
        _tex += [
            r'    \bottomrule', r'\end{tabular}',
            r'\caption{Reasoning-alignment rates (\%) of the full True Positive pools on the three backbones run with '
            r'both prompt configurations, as labeled by the classifier and corrected for its errors. The correction '
            r'weights P(human label $\mid$ classifier label), estimated on the blinded validation sample '
            r'(\Cref{tab:rq2_blind_validation}), by each pool\textquotesingle s classifier-label mix. Brackets: 95\% '
            r'bootstrap CI (10{,}000 resamples within classifier label).}\label{tab:rq2_alignment_corrected}',
            r'\end{table}',
        ]
        with open(os.path.join(REBUTTAL_DIR, 'rq2_alignment_corrected.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')

        # (d) FP adjudication. The annotator was not a clinician, so "warranted" (needed to
        # prevent harm) and "supplementary" are merged into one "reasonable" category.
        _FP_CATEGORY = {
            'warranted': 'Reasonable; requires a clinician to decide',
            'supplementary': 'Reasonable; requires a clinician to decide',
            'already_addressed': 'Already addressed in the response',
            'not_warranted': 'Misreads question',
        }
        # Summarized examples (key-file items B199/B008, B164, B015). Re-check them whenever
        # the FP labels change: each must still carry the category it illustrates.
        _FP_EXAMPLE = {
            'Reasonable; requires a clinician to decide': (
                'The patient asks about therapy while describing alcohol use at home; the Judge asks the response '
                'to address drinking quantity or dependence, home safety, and addiction resources.'
            ),
            'Already addressed in the response': (
                'The Judge says the response omits the pregnancy risks of marijuana (developmental delay, low birth '
                'weight, prematurity), but the response lists all of them.'
            ),
            'Misreads question': (
                'The Judge asks for a ``do not stop your medication without guidance\'\' warning on a question about '
                'coffee, where neither the question nor the response involves medication.'
            ),
        }
        _fp_cat = _fp.assign(category=_fp['human_label'].map(_FP_CATEGORY)).dropna(subset=['category'])
        _cat_order = list(_FP_EXAMPLE)
        _counts = _fp_cat['category'].value_counts().reindex(_cat_order, fill_value=0)
        _n_fp = int(_counts.sum())
        _tex = [
            r'\begin{table*}[t]', r'\centering', r'\small',
            r'\begin{tabular}{p{0.24\linewidth}rp{0.6\linewidth}}', r'    \toprule',
            r'    \textbf{Category} & \textbf{$n$ (\%)} & \textbf{Example (summarized)} \\', r'    \midrule',
        ]
        for _cat in _cat_order:
            _tex.append(f"    {_cat} & {_counts[_cat]} ({100 * _counts[_cat] / _n_fp:.0f}\\%) & {_FP_EXAMPLE[_cat]} \\\\")
        _tex += [r'    \bottomrule', r'\end{tabular}', r'\vspace{0.6em}', '']
        # Counts by definition and backbone.
        _by = pd.crosstab([_fp_cat['Metric'], _fp_cat['Eval Model']], _fp_cat['category']).reindex(columns=_cat_order, fill_value=0)
        _tex += [
            r'\begin{tabular}{llccc}', r'    \toprule',
            r'    \textbf{Definition} & \textbf{Judge} & \textbf{Reasonable} & \textbf{Already addressed} & \textbf{Misreads question} \\',
            r'    \midrule',
        ]
        for _mi, _metric in enumerate([REBUTTAL_OH, REBUTTAL_CC]):
            _sub = _by.loc[_metric] if _metric in _by.index.get_level_values(0) else pd.DataFrame()
            for _i, (_judge, _r) in enumerate(_sub.iterrows()):
                _label = r'\multirow{' + str(len(_sub)) + r'}{*}{' + _SHORT[_metric] + r'}' if _i == 0 else ''
                _tex.append(f"    {_label} & {_judge} & " + ' & '.join(str(int(_r[c])) for c in _cat_order) + r' \\')
            if _mi == 0:
                _tex.append(r'    \midrule')
        _tex += [
            r'    \bottomrule', r'\end{tabular}',
            r'\caption{Adjudication of ' + str(_n_fp) + r' concerns raised only by the Judge (clinician: Complete; '
            r'Judge: Incomplete), sampled 10 per definition $\times$ backbone. The annotator, who is not a '
            r'clinician, judged whether the Judge\textquotesingle s main flagged omission was already addressed in '
            r'the response, misread the question, or was a reasonable concern whose necessity a clinician '
            r'would have to decide. Top: overall distribution with a summarized example. Bottom: counts by '
            r'prompt configuration and backbone.}\label{tab:fp_adjudication}',
            r'\end{table*}',
        ]
        with open(os.path.join(REBUTTAL_DIR, 'fp_adjudication.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')
        _fp_cat.groupby(['Metric', 'Eval Model', 'category']).size().rename('n').reset_index().to_csv(
            os.path.join(REBUTTAL_DIR, 'fp_adjudication_categories.csv'), index=False
        )

        blind_scores = {
            'classifier_agreement': _classifier_agreement_df, 'retest': _retest_agreement, 'confusion': _confusion_df,
            'human_matched': _human_matched_df, 'corrected_rates': _corrected_rates_df,
            'fp_adjudication': _fp_adjudication_df, 'fp_vs_code': _fp_vs_code_df, 'fp_vs_d1': _fp_vs_d1_df,
            # Classifier and human labels of the labeled TP validation items, for the per-row
            # correction in the combined alignment table.
            'tp_validation_labels': _tp_lab[['item_id', 'Metric', 'Eval Model', 'classifier_label', 'human_label']],
            'n_labeled': int(_main['human_label'].notna().sum()), 'n_items': len(_main),
        }
        for _name, _obj in blind_scores.items():
            if isinstance(_obj, pd.DataFrame):
                _obj.to_csv(os.path.join(REBUTTAL_DIR, f'blind_validation_{_name}.csv'))
        _out = mo.vstack([
            mo.md(f"**{blind_scores['n_labeled']} / {blind_scores['n_items']} items labeled.** "
                  f"Test–retest: n={_retest_agreement['n']}, agreement={_retest_agreement['agreement']:.2f}, "
                  f"α={_retest_agreement['alpha']:.2f}"),
            mo.md('**Classifier vs. human (TP alignment)**'), mo.ui.table(_classifier_agreement_df.round(3), selection=None),
            mo.md('**Confusion (rows: classifier; columns: human)**'), mo.ui.table(_confusion_df.reset_index(), selection=None),
            mo.md('**Human-label matched-pair comparison**'), mo.ui.table(_human_matched_df.round(2), selection=None),
            mo.md('**Misclassification-corrected Yes rate**'), mo.ui.table(_corrected_rates_df.round(1), selection=None),
            mo.md('**FP adjudication**'), mo.ui.table(_fp_adjudication_df.reset_index(), selection=None),
            mo.md('**FP adjudication vs. classifier FP code**'), mo.ui.table(_fp_vs_code_df.reset_index(), selection=None),
            mo.md('**FP adjudication vs. d1 label** (rows: d1 label; columns: human category)'),
            mo.ui.table(_fp_vs_d1_df.reset_index(), selection=None),
        ])
    _out
    return (blind_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combined reasoning-alignment table

    One table in place of `alignment_by_rubric` and `rq2_alignment_corrected` in the paper
    (both are still written by this notebook); written to `alignment_combined.tex`. Each row of
    `alignment_by_rubric` gets its classifier rates and the rates corrected for classifier error,
    with an arrow marking whether the correction raised or lowered the rate. The matched-pair
    comparison stays in `alignment_matched_pairs`.

    The correction weights P(human label | classifier label), estimated on the blinded TP
    validation items, by the row's classifier-label mix. Its CI comes from a joint bootstrap:
    each draw resamples the row's questions (cluster bootstrap, same draws as the classifier
    CIs in `alignment_by_rubric`) and the validation items within classifier label. It is
    therefore somewhat wider than the `rq2_alignment_corrected` CI, which holds the pool mix fixed.
    """)
    return


@app.cell
def _(
    ALIGNMENT_ROW_POOLS,
    OUTPUT_DIR,
    REBUTTAL_DIR,
    SHARED_BACKBONES,
    alignment_by_rubric_df,
    blind_scores,
    make_cluster_sampler,
    mo,
    np,
    os,
    pd,
    rebuttal_tp_df,
):
    if not blind_scores or 'tp_validation_labels' not in blind_scores or alignment_by_rubric_df.empty:
        alignment_combined_df = pd.DataFrame()
        _out = mo.callout(mo.md('No TP validation labels or alignment rows; skipping the combined table.'), kind='info')
    else:
        _ALIGN = ['Yes', 'Partially', 'No']
        # Reported outcomes: the three labels and Yes or Partially (any shared reasoning).
        _OUTCOMES = {'Yes': [0], 'Partially': [1], 'No': [2], 'Yes+Partially': [0, 1]}
        _N_RESAMPLES = 10000
        _val = blind_scores['tp_validation_labels']
        _val_cls = _val['classifier_label'].values
        _val_hum = _val['human_label'].values
        _validated_backbones = set(_val['Eval Model'])

        def _conditional(cls, hum):
            """3x3 matrix P(human = h | classifier = c); rows c, columns h, in _ALIGN order."""
            return np.array([
                [(hum[cls == c] == h).mean() if (cls == c).any() else np.nan for h in _ALIGN]
                for c in _ALIGN
            ])

        # Validation resamples, shared by every row so all rows use the same draws.
        _strata = [np.flatnonzero(_val_cls == c) for c in _ALIGN]
        assert all(len(s) for s in _strata), 'a classifier label has no validation items'
        _P = _conditional(_val_cls, _val_hum)
        _rng_val = np.random.default_rng(20261011)
        _P_boot = np.empty((_N_RESAMPLES, 3, 3))
        for _b in range(_N_RESAMPLES):
            _idx = np.concatenate([_rng_val.choice(s, size=len(s), replace=True) for s in _strata])
            _P_boot[_b] = _conditional(_val_cls[_idx], _val_hum[_idx])

        _classifier = alignment_by_rubric_df.set_index(['metric', 'eval_model'])
        _rows = []
        # The last row pools both definitions on the shared backbones (the pooled rate the
        # paper quotes for both rubrics together, formerly in rq2_alignment_corrected).
        _BOTH = 'Both definitions'
        _row_specs = list(ALIGNMENT_ROW_POOLS) + [(_BOTH, 'Shared', SHARED_BACKBONES)]
        for _ri, (_rubric, _key, _members) in enumerate(_row_specs):
            _metric_sel = (
                rebuttal_tp_df['Metric'].isin(alignment_by_rubric_df['metric'].unique()) if _rubric == _BOTH
                else rebuttal_tp_df['Metric'] == _rubric
            )
            _pool = rebuttal_tp_df[_metric_sel & rebuttal_tp_df['Eval Model'].isin(_members)]
            _lab = _pool['aligned'].values
            _mix = np.array([(_lab == c).mean() for c in _ALIGN])
            _c = _classifier.loc[(_rubric, _key)] if _rubric != _BOTH else None
            if _c is not None:
                assert len(_pool) == _c['n_pairs'] and np.allclose(_mix, [_c[a] for a in _ALIGN]), (_rubric, _key)
            # Same seed and draws as alignment_by_rubric, so the classifier CIs match that table.
            _draw = make_cluster_sampler(_pool['qid'].values)
            _rng = np.random.default_rng(20261114 + _ri)
            _cls_boot = np.empty((_N_RESAMPLES, 3))
            _corr_boot = np.empty((_N_RESAMPLES, 3))
            for _b in range(_N_RESAMPLES):
                _idx = _draw(_rng)
                _cls_boot[_b] = [(_lab[_idx] == c).mean() for c in _ALIGN]
                _corr_boot[_b] = _cls_boot[_b] @ _P_boot[_b]
            _corr = _mix @ _P
            _row = {
                'metric': _rubric, 'eval_model': _key, 'n_pairs': len(_pool),
                # The validation sample covers only the shared backbones; rows with other
                # backbones extrapolate the correction to them.
                'extrapolated': not set(_members) <= _validated_backbones,
            }
            for _name, _cols in _OUTCOMES.items():
                _cb = _cls_boot[:, _cols].sum(axis=1)
                _kb = _corr_boot[:, _cols].sum(axis=1)
                _row.update({
                    _name: _mix[_cols].sum(),
                    f'{_name}_lo': np.percentile(_cb, 2.5), f'{_name}_hi': np.percentile(_cb, 97.5),
                    f'{_name}_corr': _corr[_cols].sum(),
                    f'{_name}_corr_lo': np.nanpercentile(_kb, 2.5), f'{_name}_corr_hi': np.nanpercentile(_kb, 97.5),
                })
            for _a in _ALIGN if _c is not None else []:
                assert np.isclose(_row[f'{_a}_lo'], _c[f'{_a}_lo']) and np.isclose(_row[f'{_a}_hi'], _c[f'{_a}_hi']), (_rubric, _key, _a)
            _rows.append(_row)
        alignment_combined_df = pd.DataFrame(_rows)
        alignment_combined_df.to_csv(os.path.join(REBUTTAL_DIR, 'alignment_combined.csv'), index=False)

        # ── LaTeX ──────────────────────────────────────────────────────────────
        def _cell(v, lo, hi, ref=None):
            """Value over its CI; with ref, an arrow marks a rise or fall against ref (at one decimal)."""
            if pd.isna(v):
                return '--'
            _arrow = ''
            if ref is not None and round(100 * v, 1) != round(100 * ref, 1):
                _arrow = r'$\uparrow$' if v > ref else r'$\downarrow$'
            return (r'\shortstack{' + f'{100 * v:.1f}' + _arrow + r' \\ {\scriptsize ['
                    + f'{100 * lo:.1f}, {100 * hi:.1f}' + r']}}')

        _JUDGE_TEX = {
            'All': r'\textit{All}',
            'All-excl-OpenBioLLM': r'\textit{All excl. OpenBioLLM}',
            'Shared': r'\textit{Shared backbones}',
            'OpenBioLLM': r'OpenBioLLM$^*$',
        }
        _k = len(_OUTCOMES)
        # Two-line header for the joint outcome keeps its column as narrow as its values.
        _head = ' & '.join(r'\textbf{\shortstack{Yes+ \\ Partially}}' if o == 'Yes+Partially' else r'\textbf{' + o + r'}' for o in _OUTCOMES)
        _tex = [
            r'\begin{table*}[t]',
            r'\centering',
            r'\small',
            r'\setlength{\tabcolsep}{3.5pt}',
            r'\renewcommand{\arraystretch}{1.1}',
            r'\begin{tabular}{lr' + 'c' * (2 * _k) + r'}',
            r'    \toprule',
            r'    & & \multicolumn{' + str(_k) + r'}{c}{\textbf{Classifier labels}} & \multicolumn{' + str(_k)
            + r'}{c}{\textbf{Corrected for classifier error}} \\',
            r'    \cmidrule(lr){3-' + str(2 + _k) + r'} \cmidrule(lr){' + str(3 + _k) + '-' + str(2 + 2 * _k) + r'}',
            r'    \textbf{Judge} & $\boldsymbol{n}$ & ' + _head + ' & ' + _head + r' \\',
        ]
        for _rubric in alignment_combined_df['metric'].unique():
            _sub = alignment_combined_df[alignment_combined_df['metric'] == _rubric]
            _tex += [r'    \midrule', r'    \multicolumn{' + str(2 + 2 * _k) + r'}{l}{\textit{' + _rubric + r'}} \\']
            _first_pooled = True
            for _, _r in _sub.iterrows():
                if _r['eval_model'] in ('All', 'All-excl-OpenBioLLM', 'Shared') and _first_pooled and len(_sub) > 1:
                    _tex.append(r'    \cmidrule(l){1-' + str(2 + 2 * _k) + r'}')
                    _first_pooled = False
                _judge = _JUDGE_TEX.get(_r['eval_model'], _r['eval_model'])
                _dag = r'$^\dagger$' if _r['extrapolated'] else ''
                _tex.append(
                    f"    \\quad {_judge}{_dag} & {int(_r['n_pairs'])} & "
                    + ' & '.join(_cell(_r[o], _r[f'{o}_lo'], _r[f'{o}_hi']) for o in _OUTCOMES) + ' & '
                    + ' & '.join(_cell(_r[f'{o}_corr'], _r[f'{o}_corr_lo'], _r[f'{o}_corr_hi'], ref=_r[o]) for o in _OUTCOMES)
                    + r' \\'
                )
        _tex += [
            r'    \bottomrule',
            r'\end{tabular}',
            r'\caption{Reasoning alignment of the True Positive (TP) pairs (clinician and Judge both rated the '
            r'response Incomplete) on MedExpert, as \% of the $n$ pairs. \textit{Yes}: the Judge cited every clinician '
            r'concern; \textit{Partially}: at least one, at least in part; \textit{No}: none; \textit{Yes+Partially}: '
            r'any shared reasoning. \textit{Classifier labels} are the GPT-5.4~Mini labels. \textit{Corrected} rates '
            r'weight P(human label $\mid$ classifier label), estimated on the '
            f'{len(_val)}'
            r' blinded validation pairs (\Cref{tab:rq2_blind_validation}), by the row\textquotesingle s classifier-label '
            r'mix; $\uparrow$/$\downarrow$: the correction raised/lowered the rate. Brackets: 95\% bootstrap CIs '
            r'(' + f'{_N_RESAMPLES:,}'.replace(',', '{,}') + r' resamples of the questions; for corrected rates, also of '
            r'the validation pairs within classifier label). $^\dagger$The validation pairs come from the shared '
            r'backbones only, so the correction is extrapolated to this row. \textit{All} pools every backbone run '
            r'with the definition; \textit{Shared backbones} pools LLaMA~3.3-70B, GPT-5~Mini and Qwen~3.6, the three '
            r'backbones run with both definitions; \textit{Both definitions} pools their TP pairs under both. '
            r'$^*$OpenBioLLM-70B is excluded from the main-text analysis '
            r'(\Cref{app:openbiollm}).'
            r'}\label{tab:alignment_combined}',
            r'\end{table*}',
        ]
        # Written to OUTPUT_DIR so the latex-build copy step syncs it to the paper.
        with open(os.path.join(OUTPUT_DIR, 'alignment_combined.tex'), 'w') as _f:
            _f.write('\n'.join(_tex) + '\n')
        _out = mo.ui.table(alignment_combined_df.round(3), selection=None, page_size=len(alignment_combined_df))
    _out
    return (alignment_combined_df,)


if __name__ == "__main__":
    app.run()
