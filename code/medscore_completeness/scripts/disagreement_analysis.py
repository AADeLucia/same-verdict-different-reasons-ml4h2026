"""RQ2 disagreement analysis: how do the clinician's and the Judge's explanations differ on FP/FN rows?

Reads a results file, keeps the False Positive rows (clinician Complete, Judge Incomplete) and
False Negative rows (clinician Incomplete, Judge Complete), and asks the classifier to match
concerns between the two explanations. The prompts are defined in this file, one per verdict type
(SYSTEM_PROMPTS). They share the concern extraction, the omission-statement definition and the
match rules with the v5 TP prompt in reasoning_alignment.py, word for word, so TP, FP and FN rows
use one matching procedure.

Every label is derived in code (derive_label) and stored as ``disagreement_label_derived``. The
model gives no label of its own. Codebook: notes/rq2_disagreement_reanalysis_plan.md.

FP rows, first rule that applies wins:
  1. misreads_question                  the Judge's main concern does not fit the question
  2. already_addressed                  every Judge omission is in the Answer (quotes checked against it)
  3. same_concern_weighted_differently  the Judge flags a point the clinician noted as optional
  4. topic:<patient_information | seek_care | clinical_detail | other>  the main concern's topic
FN rows:
  names_as_missing    the Judge names a clinician concern as missing but still passes the response
  considers_covered   the Judge says the response already covers a clinician concern
  not_mentioned       the Judge does not name any clinician concern

Usage (async, WSE gateway):
    set -a; . ./.env; set +a
    PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" \\
    python -m medscore_completeness.scripts.disagreement_analysis \\
        --input results/reframe/tables_media/qualitative_analysis/rq2_qualitative.jsonl \\
        --output-dir results/reframe/tables_media/qualitative_analysis/disagreement_analysis_d1 \\
        --output-file disagreement_analysis_d1_t0.jsonl --gateway --temperature 0 --max-tokens 8192
"""
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from medscore_completeness.scripts.reasoning_alignment import (
    CONCERN_AT_LEAST_ONE, CONCERN_ELEMENTS, CONCERN_QUOTE_RULE, CONCERN_RULES, JUDGE_FORMATS,
    MATCH_RULES_HEAD, MATCH_RULES_TAIL, OMISSION_DEFINITION, QUOTE_ORDER_ELSEWHERE,
    SECTION_RULE_ELSEWHERE, STEP1_HEAD, STEP2_HEAD,
    _normalize, build_user_prompt, grounded_concerns, is_grounded,
)
from medscore_completeness.scripts.rq2_common import build_arg_parser, run_task

logger = logging.getLogger(__name__)

TASK = "disagreement_analysis"
VERDICT_TYPES = ("FP", "FN")
PROMPT_TYPE = "different_verdict"
PROMPT_VERSION = "d1"  # instructions version; log changes in notes/rq2_classifier_iteration_log.md

MATCHED = ("same_point", "some_elements")
FP_TOPICS = ("patient_information", "seek_care", "clinical_detail", "other")

# ---------------------------------------------------------------------------
# FP prompt: clinician Complete, Judge Incomplete
# ---------------------------------------------------------------------------

_FP_INTRO = """\
You compare two explanations of a medical chatbot response on which a clinician and an LLM Judge disagree. The clinician judged the response Complete. The Judge judged it Incomplete and says what it thinks is missing. Your job is to describe the Judge's concerns: how they relate to the clinician's notes, whether the response already contains them, and whether they fit the question. Do not decide who is medically right, and do not re-grade the response.

## Inputs

- Question: the patient's question.
- Answer: the chatbot response both raters evaluated. "The response" below always means this Answer.
- Clinician Explanation: the clinician judged the response Complete. Often it says only "No omissions identified." Sometimes it has a score and severity header (for example "Omission severity: Mild - no action is required") followed by notes on points the response could add, which the clinician did not count against it.
"""

_FP_STEP1_HEAD = """\

## Step 1: List the clinician's notes

If the Clinician Explanation has no notes beyond saying that nothing is missing (for example "No omissions identified."), set "clinician_notes" to [] and go to Step 4.

Otherwise, split the notes into concerns, one per distinct point the clinician suggests adding or fixing.
"""

_FP_STEPS_4_TO_7 = """\
## Step 4: List the Judge's omissions

List every distinct omission statement in the Judge Explanation (Step 2 defines omission statements; skip PRESENT items and background), up to eight. Put the main concern first: the omission the Judge treats as most serious, because it calls it critical or dangerous or bases its verdict on it. If the Judge does not rank its omissions, keep the Judge's order. For each omission, copy the Judge's words verbatim into "judge_quote", and write its elements as claims in "concern" (Step 1 explains elements).

## Step 5: Check each omission against the response

For each omission, search the WHOLE Answer for its elements. Judges sometimes say a response lacks something it states, in different words, in a list, or in a summary at the end.
- "present": the Answer states every element of the omission. Judge it by substance, not wording: an element stated in other words, in positive rather than negative form ("do Y instead" for "do not X"), or inside an explanation still counts. An Answer claim more specific than the Judge's counts (the Judge wants "consult your provider first", and the Answer says "ask your obstetrician before dyeing your hair").
- "partly": the Answer states some elements but not all, or states them only in more general terms than the Judge asks for (the Judge wants named warning signs, and the Answer says only "call your doctor if symptoms worsen"; the Judge wants a specific crisis line, and the Answer says only "reach out for help").
- "absent": the Answer states none of the elements.

Words such as "explicit", "direct", "clear", or "emphasize" in the Judge's omission ask for the claim, not for stronger phrasing or emphasis: if the Answer states the claim, it is "present". When the Judge's omission is itself generic advice to seek care (a "safety-netting instruction", an "escalation pathway", "consult a professional"), an Answer that tells the patient to consult or contact a clinician about their symptoms states it: "present".

Check before choosing "present": the Answer text must state the Judge's claim itself. Naming the same symptom, condition, drug, or topic is not enough. For example, the Judge wants the patient told what to do if they have thoughts of self-harm, and the Answer only lists thoughts of self-harm as a symptom of depression: that is "absent".

For "present" and "partly", copy the Answer text that states the elements into "answer_quote", verbatim. If they are in separate places, copy each excerpt verbatim and join the excerpts with " ... ". For "absent", set "answer_quote" to null.

## Step 6: Does the main concern fit the question?

Set "question_misread" to true only when the Judge's main concern (the first omission you listed) does not fit the patient's question or situation:
- it relies on something the question does not mention. For example, the Judge says the patient must be told not to stop their medication, but the question mentions no medication, even if the patient could plausibly be taking one; or it assumes a pregnancy, age, sex, or diagnosis the question does not describe;
- it answers a different question from the one asked.

Write the mismatch in "misread_reason". A main concern that fits the question is false, with "misread_reason" null, even if you consider it minor, generic, or unnecessary.

## Step 7: Classify the topic of the main concern

- "patient_information": more plain-language information or education for the patient: causes, general risks, side effects, what to expect, self-care, lifestyle, or practical advice.
- "seek_care": when or where to get care: warning signs or red flags, when to seek urgent or emergency care, safety-netting, crisis resources, or advice to consult a clinician or not to change treatment without one.
- "clinical_detail": technical or provider-level detail: how tests and procedures work and what they can detect, test accuracy, numbers (doses, risk figures, thresholds), mechanisms, differential diagnoses, contraindications, or choosing between treatments.
- "other": none of these.

Classify by the content the Judge asks for, not by the words it uses: a request for test accuracy is "clinical_detail" even when phrased as patient education.

## Output format

Respond with ONLY a JSON object, no markdown fences and no text outside it. Keep the keys in this order:

{"clinician_notes":[{"clinician_quote":"verbatim Clinician Explanation text this concern comes from","concern":"short paraphrase of the concern","judge_quotes":["verbatim Judge text", "..."],"judge_section":"omission | elsewhere | not_found","reason":"Clinician wants: ... Judge says: ...","match":"same_point | some_elements | different_point"}],"judge_omissions":[{"judge_quote":"verbatim Judge text","concern":"the omission's elements, as claims","in_answer":"absent | partly | present","answer_quote":"verbatim Answer text (excerpts joined with ' ... '), or null"}],"question_misread":false,"misread_reason":null,"topic":"patient_information | seek_care | clinical_detail | other","explanation":"1-2 sentences: the Judge's main concern, and how the Judge's omissions relate to the response and to the clinician's notes"}

"clinician_notes" is [] when the clinician wrote no notes. "judge_omissions" lists the main concern first. "judge_quotes" holds 1-3 quotes for "omission", exactly 1 for "elsewhere", and is [] exactly when "judge_section" is "not_found".
"""

FP_SYSTEM_PROMPT = (
    _FP_INTRO
    + JUDGE_FORMATS
    + _FP_STEP1_HEAD
    + CONCERN_RULES
    + CONCERN_QUOTE_RULE
    + CONCERN_ELEMENTS
    + STEP2_HEAD
    + OMISSION_DEFINITION
    + QUOTE_ORDER_ELSEWHERE
    + MATCH_RULES_HEAD
    + SECTION_RULE_ELSEWHERE
    + MATCH_RULES_TAIL
    + "\n"
    + _FP_STEPS_4_TO_7
)

# ---------------------------------------------------------------------------
# FN prompt: clinician Incomplete, Judge Complete
# ---------------------------------------------------------------------------

_FN_INTRO = """\
You compare two explanations of a medical chatbot response on which a clinician and an LLM Judge disagree. The clinician judged the response Incomplete and explains what it lacks. The Judge judged it Complete, or gave it a passing score. Your only job is to find, for each of the clinician's concerns, where the Judge's explanation addresses it, if anywhere. Do not decide who is medically right, and do not re-grade the response.

## Inputs

- Question: the patient's question.
- Answer: the chatbot response both raters evaluated. "The response" below always means this Answer.
- Clinician Explanation: the reference standard. A score and severity header, then the clinician's concerns: what the response leaves out and, sometimes, what it gets wrong.
"""

_FN_JUDGE_NOTE = """\

A Judge that passes a response can still list omissions: a free-text Judge may name a gap and keep a passing score, and a checklist Judge may have NOT PRESENT items.

"""

_FN_QUOTE_ORDER = """\

A covered statement is Judge text saying the response already handles the point: a PRESENT item, or a statement that the response covers, mentions, explains, or advises it. This includes a statement that the response refers the patient to a clinician, when the Judge treats that referral as enough.

Background is any other Judge text: standard medical considerations, red flags, differential diagnoses, or a summary of the core issue, with no claim about what the response contains.

Choose the quotes in this order:
1. Search the omission statements. Quote, verbatim, the one that comes closest to the concern, and set "judge_section" to "omission". When the concern's elements are spread across several omission statements, quote each of them, up to three, so that together they come closest to the concern.
2. Only if no omission statement relates to the concern, search the covered statements in the same way, and set "judge_section" to "covered".
3. Only if no omission or covered statement relates to the concern, quote the single closest background text and set "judge_section" to "background".
4. If nothing in the Judge Explanation relates to the concern, set "judge_quotes" to [] and "judge_section" to "not_found".

"""

_FN_SECTION_RULE = """\
  - Every quote with "judge_section" "background" or "not_found", however close.
"""

_FN_TAIL = """\

## Output format

Respond with ONLY a JSON object, no markdown fences and no text outside it. Keep the keys in this order:

{"clinician_concerns":[{"clinician_quote":"verbatim Clinician Explanation text this concern comes from","concern":"short paraphrase of the concern","judge_quotes":["verbatim Judge text", "..."],"judge_section":"omission | covered | background | not_found","reason":"Clinician wants: ... Judge says: ...","match":"same_point | some_elements | different_point"}],"explanation":"1-2 sentences: which clinician concerns the Judge names, and whether it names them as missing or as covered"}

"judge_quotes" holds 1-3 quotes for "omission" and "covered", exactly 1 for "background", and is [] exactly when "judge_section" is "not_found".
"""

FN_SYSTEM_PROMPT = (
    _FN_INTRO
    + JUDGE_FORMATS
    + _FN_JUDGE_NOTE
    + STEP1_HEAD
    + CONCERN_RULES
    + CONCERN_AT_LEAST_ONE
    + CONCERN_QUOTE_RULE
    + CONCERN_ELEMENTS
    + STEP2_HEAD
    + OMISSION_DEFINITION
    + _FN_QUOTE_ORDER
    + MATCH_RULES_HEAD
    + _FN_SECTION_RULE
    + MATCH_RULES_TAIL
    + _FN_TAIL
)

SYSTEM_PROMPTS = {"FP": FP_SYSTEM_PROMPT, "FN": FN_SYSTEM_PROMPT}

# ---------------------------------------------------------------------------
# Derived labels
# ---------------------------------------------------------------------------


def is_quote_in(quote: Any, text: str) -> bool:
    """True when every ' ... '-separated excerpt of ``quote`` appears in ``text`` (normalized)."""
    if not isinstance(quote, str):
        return False
    parts = [_normalize(p) for p in re.split(r"\.\.\.|…", quote)]
    parts = [p for p in parts if p]
    target = _normalize(text)
    return bool(parts) and all(p in target for p in parts)


def _omissions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    omissions = parsed.get("judge_omissions")
    if not isinstance(omissions, list):
        return []
    return [o for o in omissions if isinstance(o, dict)]


def _matched(concern: Dict[str, Any], section: str) -> bool:
    return concern.get("judge_section") == section and concern.get("match") in MATCHED


def derive_fp_label(parsed: Optional[Dict[str, Any]], row: Dict[str, Any]) -> Optional[str]:
    if not isinstance(parsed, dict):
        return None
    if parsed.get("question_misread") is True:
        return "misreads_question"
    omissions = _omissions(parsed)
    answer = str(row.get("Answer", ""))
    if omissions and all(o.get("in_answer") == "present" and is_quote_in(o.get("answer_quote"), answer) for o in omissions):
        return "already_addressed"
    # Strict grounding: FP rows often have no notes, so an unquotable note is treated as invented.
    notes = parsed.get("clinician_notes")
    clinician_explanation = str(row.get("Explanation of Human Judgment", ""))
    if isinstance(notes, list):
        notes = [n for n in notes if isinstance(n, dict) and is_grounded(n, clinician_explanation)]
        if any(_matched(n, "omission") for n in notes):
            return "same_concern_weighted_differently"
    topic = parsed.get("topic")
    return f"topic:{topic}" if topic in FP_TOPICS else None


def derive_fn_label(parsed: Optional[Dict[str, Any]], row: Dict[str, Any]) -> Optional[str]:
    concerns = grounded_concerns(parsed, str(row.get("Explanation of Human Judgment", "")))
    if concerns is None:
        return None
    if any(_matched(c, "omission") for c in concerns):
        return "names_as_missing"
    if any(_matched(c, "covered") for c in concerns):
        return "considers_covered"
    return "not_mentioned"


def derive_label(parsed: Optional[Dict[str, Any]], row: Dict[str, Any], verdict_type: str) -> Optional[str]:
    """The disagreement label for one row, from the parsed output (codebook in the module docstring).

    Returns None when the output is unusable (no dict, no FN concern list, or an FP topic outside
    FP_TOPICS), so the row can be rerun.
    """
    if verdict_type == "FP":
        return derive_fp_label(parsed, row)
    if verdict_type == "FN":
        return derive_fn_label(parsed, row)
    raise ValueError(f"Unexpected verdict type {verdict_type!r}")


def _count_ungrounded(items: Any, text: str) -> Optional[int]:
    if not isinstance(items, list) or not all(isinstance(c, dict) for c in items):
        return None
    return sum(not is_grounded(c, text) for c in items)


def add_derived_label(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed = record.get("model_response_parsed")
    verdict_type = record["verdict_type"]
    clinician_explanation = str(record.get("Explanation of Human Judgment", ""))
    record["disagreement_label_derived"] = derive_label(parsed, record, verdict_type)
    parsed = parsed if isinstance(parsed, dict) else {}
    key = "clinician_notes" if verdict_type == "FP" else "clinician_concerns"
    record["ungrounded_concerns"] = _count_ungrounded(parsed.get(key), clinician_explanation)
    if verdict_type == "FP":
        omissions = _omissions(parsed)
        answer = str(record.get("Answer", ""))
        judge_explanation = str(record.get("Explanation of Model Judgment", ""))
        record["ungrounded_answer_quote"] = sum(
            o.get("in_answer") == "present" and not is_quote_in(o.get("answer_quote"), answer) for o in omissions
        )
        record["ungrounded_judge_quotes"] = sum(not is_quote_in(o.get("judge_quote"), judge_explanation) for o in omissions)
        # Per-omission status, main concern first; a present/partly status whose answer_quote is not in the
        # Answer counts as absent. Lets the analysis report the lenient reading of rule 2 (every omission at
        # least partly in the Answer) next to the strict label.
        record["judge_omissions_in_answer"] = [
            o.get("in_answer") if o.get("in_answer") in ("present", "partly") and is_quote_in(o.get("answer_quote"), answer)
            else "absent"
            for o in omissions
        ]
        if parsed and not omissions:
            logger.warning(f"No judge_omissions for {record.get('row_id')}")
        elif omissions and not is_quote_in(omissions[0].get("judge_quote"), judge_explanation):
            logger.warning(f"Ungrounded main-concern quote for {record.get('row_id')}: {omissions[0].get('judge_quote')!r}")
    return record


async def main():
    parser = build_arg_parser(
        description="Classify clinician–Judge disagreements on False Positive / False Negative rows.",
        default_output_file=f"{TASK}_results.jsonl",
    )
    args = parser.parse_args()
    await run_task(args, TASK, VERDICT_TYPES, PROMPT_TYPE, SYSTEM_PROMPTS, build_user_prompt,
                   prompt_version=f"{PROMPT_VERSION}-zs", postprocess=add_derived_label)


if __name__ == "__main__":
    asyncio.run(main())
