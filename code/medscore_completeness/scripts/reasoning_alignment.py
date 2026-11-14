"""RQ2 reasoning alignment: does the Judge's explanation name the clinician's concerns?

Reads a results file, keeps the True Positive rows (clinician and Judge both say Incomplete;
cross-dataset 'Agrees' rows are resolved to TP/TN), and asks the classifier for a
Yes / Partially / No alignment label. The prompt is defined in this file (SYSTEM_PROMPT,
USER_TEMPLATE); --few-shot appends the examples in reasoning_alignment_examples.py. Each run
stores the full prompts and the prompt version (e.g. v5-zs, v5-fs) in its output records.

Each record also gets ``omissions_aligned_derived``: the label recomputed in code from the
per-concern matches (derive_label): any quote outside an omission statement counts as
different_point, and concerns whose clinician_quote is not in the Clinician Explanation are dropped
(counted in ``ungrounded_concerns``). ``model_response_parsed.omissions_aligned`` is the model's own label.

Usage (async):
    PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" \\
    python -m medscore_completeness.scripts.reasoning_alignment \\
        --input results/reframe/tables_media/qualitative_analysis/rq2_qualitative.jsonl \\
        --output-dir results/reframe/tables_media/qualitative_analysis

Batch API: add --batch to submit, then rerun with --resume to poll and retrieve.

Validation subset (the blinded human sample):
    ... --subset-file results/reframe/tables_media/qualitative_analysis/ml4h_blind/rq2_blind_validation_key_reduced.jsonl \\
        --output-file reasoning_alignment_dev.jsonl
"""
import asyncio
import re
from typing import Any, Dict, List, Optional

from medscore_completeness.scripts.reasoning_alignment_examples import FEW_SHOT_EXAMPLES
from medscore_completeness.scripts.rq2_common import build_arg_parser, run_task

TASK = "reasoning_alignment"
VERDICT_TYPES = ("TP",)
PROMPT_TYPE = "same_verdict"  # value notebooks/analysis.py uses to read omissions_aligned
PROMPT_VERSION = "v5"  # instructions version; log changes in notes/rq2_classifier_iteration_log.md

# The prompt is assembled from pieces so that disagreement_analysis.py can share the Judge formats,
# the concern extraction (Step 1), the omission-statement definition (Step 2) and the match rules
# (Step 3) word for word. Public names are shared; names with a leading underscore are specific to
# TP rows. The assembled text is identical to the v5 prompt (checked against the stored v5 run).

_TP_INTRO = """\
You compare two explanations of why a medical chatbot response is incomplete: one written by a clinician, one written by an LLM Judge. Both already judged the same response Incomplete. Your only job is to decide whether the Judge's explanation names the concerns the clinician raised.

## Inputs

- Question: the patient's question.
- Answer: the chatbot response both raters evaluated. "The response" below always means this Answer.
- Clinician Explanation: the reference standard. A score and severity header, then the clinician's concerns: what the response leaves out and, sometimes, what it gets wrong.
"""

JUDGE_FORMATS = """\
- Judge Explanation: the LLM Judge's reasoning, in one of two formats:
  - Free text, often in sections: the core issue; standard medical considerations, red flags, and differential diagnoses; a comparison with the response that lists its omissions; and the clinical significance.
  - A checklist the Judge wrote before reading the response, split into PRESENT (criteria the Judge thinks the response meets) and NOT PRESENT (criteria the Judge thinks it misses).
"""

_TP_MEASURE = """\

## What you measure

Recall of the clinician's concerns: did the Judge flag what the clinician flagged? Concerns the Judge raises that the clinician did not are irrelevant. They never lower the label, however many there are. Do not decide who is medically right, and do not re-grade the response. Use the Question and Answer only to understand what each rater refers to (for example, a clinician who quotes the response).

"""

STEP1_HEAD = """\
## Step 1: List the clinician's concerns

Split the Clinician Explanation into concerns, one per distinct point the clinician wants added or fixed.
"""

CONCERN_RULES = """\
- A bullet is usually one concern. Split a bullet that asks for two unrelated things (often joined by "Also" or "Can also include"). Keep a bullet together when its parts make one point (a list of warning signs, or a recommendation and its reason).
- A correction of something the response states wrongly is a concern.
- Tentative suggestions ("It might be a good idea to include...", "Could also mention...") are concerns, the same as firm ones.
- Ignore the score and severity header, and lines that only explain the severity rating or praise the response.
"""

CONCERN_AT_LEAST_ONE = """\
- Every Clinician Explanation contains at least one concern. If you find none, re-read it and list its main point.
"""

CONCERN_QUOTE_RULE = """\
- For each concern, copy the clinician's words it comes from into "clinician_quote", verbatim. Only text in the Clinician Explanation can be a concern. A point that appears only in the Judge Explanation is never a concern, however relevant it seems.
"""

CONCERN_ELEMENTS = """\

Each concern has one or more elements. An element is something the clinician wants the patient told, written as a claim: "avoid flying after 36 weeks", "anxiety can also disrupt sleep", "call 911 for heavy bleeding". A term on its own ("flying", "anxiety", "bleeding") is not an element.
"""

STEP2_HEAD = """\

## Step 2: Find the Judge's quote for each concern

Read the WHOLE Judge Explanation first. Long explanations bury the matching item in the middle of a list of ten or more.

"""

OMISSION_DEFINITION = """\
An omission statement is Judge text saying the response is missing something or gets something wrong:
- it lacks, omits, fails to mention, or does not include something;
- it is wrong or misleading, or understates or overstates something;
- an item in a NOT PRESENT list or an "omissions" section.

These are NOT omission statements, even when they name the clinician's point:
- PRESENT items, or statements that the response already covers something. Here the Judge disagrees with the clinician.
- Background the Judge lists as standard medical considerations, red flags, differential diagnoses, or a summary of the core issue, without saying the response lacks it.

In a free-text Judge Explanation with numbered or titled sections, the section decides, not the wording. Everything in a section about the core issue, standard medical considerations, red flags, or differential diagnoses is background, even when it is phrased as an instruction ("give guidance on limits", "should mention X", "advise..."). In such explanations, omission statements come only from the section that compares the response with those considerations, or from text that explicitly says the response lacks or gets wrong something. The same point often appears in both places; only the version that says the response lacks it counts.
"""

QUOTE_ORDER_ELSEWHERE = """\

Choose the quotes in this order:
1. Search the omission statements only. Quote, verbatim, the one that comes closest to the concern, and set "judge_section" to "omission". When the concern's elements are spread across several omission statements (for example, one item names the emergency number and another the crisis text line), quote each of them, up to three, so that together they come closest to the concern.
2. Only if no omission statement relates to the concern, quote the single closest text from anywhere else in the Judge Explanation and set "judge_section" to "elsewhere".
3. If nothing in the Judge Explanation relates to the concern, set "judge_quotes" to [] and "judge_section" to "not_found".

"""

MATCH_RULES_HEAD = """\
## Step 3: Write the reason, then rate the match

Write "reason" before choosing "match", in this form:
"Clinician wants: <the concern's elements, as claims>. Judge says: <the claims in judge_quotes, or 'nothing on this'>."

Then compare the claims. When there are several quotes, judge them together.

- "same_point": the Judge's claims state every element of the concern. Different wording, a synonym, or a brand versus generic drug name still counts. A match can sit inside a longer item.
- "some_elements": the Judge's claims state at least one element of the concern, but not all. For example, the clinician lists four warning signs and the Judge names one, or the clinician wants a recommendation and its reason and the Judge gives only the recommendation.
- "different_point": the Judge's claims state none of the concern's elements. This includes:
  - Shared term, different claim: the same drug, condition, body system, or word used to say something else (the clinician wants the drug's interaction with alcohol, the Judge wants its withdrawal symptoms; both mention "the placenta", but the clinician's point is what tissue is sampled and the Judge's is result accuracy).
  - Judge more general than the clinician: the clinician asks for something specific (call emergency services or a crisis line, a named treatment, particular warning signs, a specific reason) and the Judge says only something generic ("lacks safety-netting", "should seek help if symptoms worsen", "does not explain why this is urgent", "consult a provider").
  - Contradiction: the Judge recommends the opposite of what the clinician wants.
"""

SECTION_RULE_ELSEWHERE = """\
  - Every quote with "judge_section" "elsewhere" or "not_found", however close.
"""

MATCH_RULES_TAIL = """\

Specificity: a Judge claim MORE specific than the clinician's (adds week cutoffs, names the drug) can be "same_point". A Judge claim MORE general than the clinician's is "different_point".

Exception for generic escalation: when the clinician's concern is itself generic advice to seek care sooner, a generic escalation statement from the Judge ("no guidance on when to seek urgent care") is "same_point".

Check before choosing "same_point" or "some_elements": the "Judge says" claim must restate a "Clinician wants" claim. If the two only share a word or a topic, the match is "different_point".
"""

_TP_TAIL = """\

## Step 4: Assign the label

- "Yes": every concern is "same_point".
- "Partially": not Yes, and at least one concern is "same_point" or "some_elements".
- "No": every concern is "different_point".

Apply these rules exactly. Do not adjust the label for how many extra concerns the Judge raised, how long its explanation is, or how close a "different_point" match felt.

## Output format

Respond with ONLY a JSON object, no markdown fences and no text outside it. Keep the keys in this order:

{"clinician_concerns":[{"clinician_quote":"verbatim Clinician Explanation text this concern comes from","concern":"short paraphrase of the concern","judge_quotes":["verbatim Judge text", "..."],"judge_section":"omission | elsewhere | not_found","reason":"Clinician wants: ... Judge says: ...","match":"same_point | some_elements | different_point"}],"omissions_aligned":"Yes | Partially | No","explanation":"1-2 sentences: which clinician concerns the Judge captured and which it missed"}

"judge_quotes" holds 1-3 quotes for "omission", exactly 1 for "elsewhere", and is [] exactly when "judge_section" is "not_found".
"""

SYSTEM_PROMPT = (
    _TP_INTRO
    + JUDGE_FORMATS
    + _TP_MEASURE
    + STEP1_HEAD
    + CONCERN_RULES
    + CONCERN_AT_LEAST_ONE
    + CONCERN_QUOTE_RULE
    + CONCERN_ELEMENTS
    + STEP2_HEAD
    + OMISSION_DEFINITION
    + QUOTE_ORDER_ELSEWHERE
    + MATCH_RULES_HEAD
    + SECTION_RULE_ELSEWHERE
    + MATCH_RULES_TAIL
    + _TP_TAIL
)

USER_TEMPLATE = """\
Compare the explanations below and return ONLY the JSON object.

Question: {question}
Answer: {answer}
Clinician Explanation: {clinician_explanation}
Judge Explanation: {judge_explanation}"""


def build_user_prompt(row: Dict[str, Any], verdict_type: str) -> str:
    """Fill USER_TEMPLATE from one results row. ``verdict_type`` is not shown (also used for FP/FN rows)."""
    return USER_TEMPLATE.format(
        question=row.get("Question", ""),
        answer=row.get("Answer", ""),
        clinician_explanation=row.get("Explanation of Human Judgment", ""),
        judge_explanation=row.get("Explanation of Model Judgment", ""),
    )


def _normalize(text: str) -> str:
    """Lowercase and collapse everything but letters and digits, for lenient verbatim checks."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_grounded(concern: Dict[str, Any], clinician_explanation: str) -> bool:
    """True when the concern's clinician_quote appears in the Clinician Explanation."""
    quote = concern.get("clinician_quote")
    if not isinstance(quote, str) or not _normalize(quote):
        return False
    return _normalize(quote) in _normalize(clinician_explanation)


def grounded_concerns(
    parsed: Optional[Dict[str, Any]],
    clinician_explanation: Optional[str],
    key: str = "clinician_concerns",
) -> Optional[List[Dict]]:
    """The usable concern list: concerns whose clinician_quote is in the Clinician Explanation.

    Without a clinician_explanation, or when no concern is grounded (a quoting failure rather
    than invented concerns), every concern is kept. Returns None for an unusable output,
    including an empty list. ``key`` names the list in ``parsed`` (disagreement_analysis.py
    uses "clinician_notes" for FP rows).
    """
    if not isinstance(parsed, dict):
        return None
    concerns = parsed.get(key)
    if not isinstance(concerns, list) or not concerns or not all(isinstance(c, dict) for c in concerns):
        return None
    if clinician_explanation is None:
        return concerns
    grounded = [c for c in concerns if is_grounded(c, clinician_explanation)]
    return grounded or concerns


def derive_label(parsed: Optional[Dict[str, Any]], clinician_explanation: Optional[str] = None) -> Optional[str]:
    """Recompute Yes / Partially / No from the per-concern matches (Step 4 of the prompt).

    Enforces two rules the model sometimes breaks: a quote that is not an omission statement
    counts as different_point whatever match the model gave it, and a concern not quoted from
    the Clinician Explanation is dropped (see grounded_concerns). Returns None when the output
    has no usable concern list.
    """
    concerns = grounded_concerns(parsed, clinician_explanation)
    if concerns is None:
        return None
    matches = [c.get("match") if c.get("judge_section") == "omission" else "different_point" for c in concerns]
    if all(m == "same_point" for m in matches):
        return "Yes"
    if any(m in ("same_point", "some_elements") for m in matches):
        return "Partially"
    return "No"


def add_derived_label(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed = record.get("model_response_parsed")
    clinician_explanation = str(record.get("Explanation of Human Judgment", ""))
    record["omissions_aligned_derived"] = derive_label(parsed, clinician_explanation)
    usable = grounded_concerns(parsed, None)
    record["ungrounded_concerns"] = (
        None if usable is None else sum(not is_grounded(c, clinician_explanation) for c in usable)
    )
    return record


async def main():
    parser = build_arg_parser(
        description="Classify Judge–clinician reasoning alignment on True Positive rows.",
        default_output_file=f"{TASK}_results.jsonl",
    )
    parser.add_argument("--few-shot", action="store_true",
                        help="Append the few-shot examples (reasoning_alignment_examples.py) to the system prompt")
    args = parser.parse_args()
    system_prompt = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}" if args.few_shot else SYSTEM_PROMPT
    version = f"{PROMPT_VERSION}-{'fs' if args.few_shot else 'zs'}"
    await run_task(args, TASK, VERDICT_TYPES, PROMPT_TYPE, system_prompt, build_user_prompt,
                   prompt_version=version, postprocess=add_derived_label)


if __name__ == "__main__":
    asyncio.run(main())
