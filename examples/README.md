# Example graded outputs

Ten graded MedExpert records per LLM Judge, illustrating the output format of
`grade_responses`. These are a sample for inspecting the schema — they are **not**
sufficient to regenerate the paper's figures and tables, which need the full graded set
(see the root README). The full graded outputs are in `results/` (Git LFS, gzipped).

Each line is one evaluated response:

| Field | Description |
|---|---|
| `id` | SHA hash identifying the record |
| `meta` | Source record: `question`, `answer`, `criteria`, `omissions`, and nested `meta` with `omission_severity`, `domain`, `human_completeness_score` |
| `completeness-human` | Clinician annotation |
| `completeness-medexpert-fss` | Omission-Harm Rubric (few-shot) |
| `completeness-medexpert` | Omission-Harm Rubric (zero-shot) |
| `completeness-icl-healthbench` | Criteria-Checklist |
| `completeness-medexpert-prometheus` | Omission-Harm Rubric, Prometheus prompt format |
| `icl_examples` | Retrieved HealthBench neighbours used to generate criteria |

Each scoring object holds `score`, `total_points`, `scoring_model`, and a `meta` list of
per-criterion records (`criterion`, `points`, `explanation`, `criteria_score`).
`criteria_score` is an integer for the scaled rubrics and a boolean for the
Criteria-Checklist. Prometheus-2 7B has no `completeness-icl-healthbench` field: it is
excluded from the Criteria-Checklist, as reported in the paper.
