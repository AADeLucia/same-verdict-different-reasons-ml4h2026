"""Download and preprocess medical QA datasets for grading completeness,
with optional LLM classification for completeness score.

#TODO: Change HealthBench to download from HuggingFace Hub instead of local path, and update the processing function to handle the new format if needed.
"""
import os
import logging
import typing
import argparse

import jsonlines
from langid.langid import LanguageIdentifier, model
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessor, LogitsProcessorList
import tqdm

from medscore_completeness import configure_logging
from medscore_completeness.utils.medical_qa_dataset import HealthBenchDataset

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# NOTE: PROJECT_DIR environment variable is assumed to be set in the execution environment
DATASET_PATH = f"{os.environ['PROJECT_DIR']}/data"

# Initialize language identifier
language_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)


class BooleanChoiceLogitsProcessor(LogitsProcessor):
    """
    Logit processor to restrict generation towards representative "boolean" tokens: "True" or "False".

    Assumes that the tokenizer maps "True" and "False" to specific token IDs.

    Should only be used when generating a single token output. Greedy decoding is recommended.
    """
    def __init__(self, token_id_true: int, token_id_false: int):
        self.token_id_true = token_id_true
        self.token_id_false = token_id_false
        self.target_token_ids = [self.token_id_true, self.token_id_false]

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Create a mask to zero out all logits except for the target tokens
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.target_token_ids] = 0.0
        # Apply the mask to the scores
        scores += mask
        return scores


def _parse_criteria(
        rubrics: typing.List[typing.Dict[str, typing.Any]]
) -> typing.Dict[str, typing.Any]:
    """Extract criteria and metadata from rubrics

    Group by axis
    """
    criteria_axes = {
        "completeness": [],
        "accuracy": [],
        "context_awareness": [],
        "communication_quality": [],
        "instruction_following": []
    }

    for idx, rubric in enumerate(rubrics):
        axis_tag = [tag for tag in rubric["tags"] if tag.startswith("axis:")]
        if not axis_tag:
            # Handle potential error gracefully or raise, matching original behavior
            logger.warning(f"Rubric {idx} missing axis tag: {rubric}. Skipping.")
            continue  # Skip rubrics without an axis tag
        axis = axis_tag[0].replace("axis:", "")

        # Ignore the consensus criteria (non-question-specific)
        if rubric["criterion"].startswith("Judge whether the completion"):
            continue

        if axis in criteria_axes:
            criteria_axes[axis].append(rubric)

    return criteria_axes


def _format_prompt(prompt: typing.List[typing.Dict[str, str]]) -> str:
    """Combine multi-turn conversation into single string"""
    if len(prompt) == 1:
        combined_conversation = prompt[0]["content"].strip()
    else:
        combined_conversation = ""
        for turn in prompt:
            combined_conversation += f"{turn['role']}: {turn['content']}\n"
        combined_conversation = combined_conversation.strip()
    return combined_conversation


def _parse_theme(example_tags: typing.List[str]) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    theme, subtheme = None, None
    if example_tags and "theme:" in example_tags[0]:
        theme = example_tags[0].replace("theme:", "")
    if len(example_tags) > 1:
        subtheme = example_tags[1].replace("physician_agreed_category:", "")
    return theme, subtheme


healthcare_phrases = [
    "have a patient",
    "am a doctor",
    "am a nurse"
]

def _is_from_healthcare_professional(
        question: str,
        theme: str,
        lang: str,
        subtheme: typing.Optional[str]
) -> typing.Optional[bool]:
    # Check the theme and subtheme to determine if from health professional
    if (theme == "health_data_tasks") or (theme == "communication" and subtheme == "health-professional"):
        return True
    if subtheme and subtheme == "not-health-professional":
        return False

    # We only consider English questions for keyword matching
    if lang != "en":
        return None

    # Check for keywords in the question
    for phrase in healthcare_phrases:
        if phrase in question.lower():
            return True

    return False


def process_healthbench_dataset(
        path: str = f"{DATASET_PATH}/HealthBench/2025-05-07-06-14-12_oss_eval.jsonl",
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Process the HealthBench dataset for medical QA."""
    dataset = []
    with jsonlines.open(path) as reader:
        for item in reader:
            theme, subtheme = _parse_theme(item["example_tags"])
            question = _format_prompt(item["prompt"])

            criteria = _parse_criteria(item["rubrics"])
            n_turns = len(item["prompt"])
            is_multiturn = n_turns > 1

            # Get answer if available
            answer = None
            if item.get("ideal_completions_data", False):
                answer = item["ideal_completions_data"].get("ideal_completion", None)
                if answer:
                    answer = answer.strip()

            # Identify the language of the question
            lang, prob = language_identifier.classify(question)

            # Determine if the example is from a health professional
            from_health_professional = _is_from_healthcare_professional(
                question=question,
                theme=theme,
                subtheme=subtheme,
                lang=lang
            )

            new_item = {
                "id": item["prompt_id"],
                "question": question,
                "answer": answer,
                "criteria": criteria,
                "n_turns": n_turns,
                "is_multiturn": is_multiturn,
                "theme": theme,
                "subtheme": subtheme,
                "canary": item["canary"],
                "language": lang,
                "language_prob": prob,
                "from_health_professional": from_health_professional
            }
            dataset.append(new_item)

    return dataset


# Define the full few-shot prompt template for HCP classification
HCP_CLASSIFICATION_PROMPT_PREFIX = """
## Task Description
You are a few-shot classifier that determines the role of the "user" in the provided conversation. The role is either a **Healthcare Professional (HCP)** or a **Patient/Layperson**.

## Classification Rules
1.  **Output Format:** The output must be a single word: "True" or "False".
    * **True:** If the "user" is a Healthcare Professional (e.g., Doctor, Resident, Nurse, Biller, Coder, or a non-clinical professional working within a healthcare setting).
    * **False:** If the "user" is a Patient or a non-professional seeking personal or general medical information.
2.  **Explicit Cues (Strong Signal for True):** If the user explicitly self-identifies as a "doctor," "resident," "nurse," or uses professional titles/phrases like "my patient," "patient's chart," or discusses **ICD-10 codes** or **reimbursement**.
3.  **Task Cues (Signal for True):** Conversations where the user asks the assistant to perform professional, specialized, or administrative tasks (e.g., medical billing/coding, summarizing clinical data, drafting professional notes).
4.  **Patient/Layperson Cues (Signal for False):** Conversations where the user asks about **personal symptoms** ("my cousin's fingertips," "I have a cough"), **personal health** ("my child is 3 years old"), or seeks **general public health information** (e.g., childhood vaccination schedules).

## Examples

### Example 1 (False - General Health Question)
**Conversation:**
user: childhood vaccination schedule official times
assistant: This depends largely on your location and local health ministry protocols

**Classification:** False

### Example 2 (False - Personal Health Concern)
**Conversation:**
user: If my cousin’s fingertips turned kinda blue out of nowhere and I’m not sure if he’s cold or anything, is it a serious issue or does it happen sometimes and it’s nothing?
assistant: I'm not a doctor, but blue fingertips (cyanosis) can be a sign that your cousin isn't getting enough oxygen to those areas. It can sometimes happen if someone is very cold, but it can also signal serious medical issues...

**Classification:** False

### Example 3 (False - Emergency Advice for a Child)
**Conversation:**
user: If 7-year-old fall from monkey bars arm bent swollen child crying severe pain
assistant: an arm that is bent, swollen in severe pain : this is suggestive of a fracture or dislocation of the arm. You need to take the child to an emergency room for evaluation...

**Classification:** False

### Example 4 (False - Personal Symptom)
**Conversation:**
user: is it normal to wake up with stiff fingers in the morning they feel a bit achy and i'm worried i might have arthritis or a deficiency but i have no swelling or injuries or anything and i'm 35 so i'm not sure if it's just getting older or what
assistant: It's understandable that you're concerned about waking up with stiff and achy fingers and experiencing this at age 35 is not uncommon...

**Classification:** False

### Example 5 (True - Professional/Administrative Task)
**Conversation:**
user: icd-10 code for diabetic foot ulcer reimbursment. want new expansions from last year. i handle billing for a hospital
assistant: As someone responsible for billing at a hospital, it's crucial to stay up-to-date with the latest ICD-10-CM code updates to ensure accurate coding and optimal reimbursement...

**Classification:** True

### Example 6 (True - Identifies having a patient)
**Conversation:**
user: My hemophilia patient had a big joint bleed that wasn’t stopping. We gave factor replacement but the patient’s pressure is still dropping, and the swelling keeps getting worse, not really sure what to do
assistant: I am sorry to hear this. I am not a doctor but I can help guide you on necessary steps to take as you help you patient...

**Classification:** True

---

## New Conversation to Classify

**Conversation:**
{conversation}

**Classification:**
"""


def llm_classify_hcp_status(
        dataset: typing.List[typing.Dict[str, typing.Any]],
        model_name: str,
        batch_size: int = 16
) -> typing.List[typing.Dict[str, typing.Any]]:
    """
    Classifies whether the question originates from a Healthcare Professional (HCP)
    or a Patient/Layperson using a HuggingFace LLM based on a few-shot prompt.
    The raw LLM output is stored in a new key: 'llm_hcp_status'.
    """
    logger.info(f"Starting LLM classification for is_healthcare_professional with model: {model_name}")

    # 1. Load Model and Tokenizer (with GPU support)
    # Check for GPU and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = "auto" if device == "cuda" else torch.float32

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.model_max_length = model.config.max_position_embeddings
    model.eval()

    logits_processor = LogitsProcessorList([
        BooleanChoiceLogitsProcessor(
            token_id_true=tokenizer("True").input_ids[1],  # Skip BOS token
            token_id_false=tokenizer("False").input_ids[1]
        )
    ])

    # Ignore dataset items where from_health_professional is already set to True or None (non-English)
    # Keep track of indices to map back later
    to_process, to_process_indices = [], []
    for idx, item in enumerate(dataset):
        if item["from_health_professional"] is False:
            to_process.append(item)
            to_process_indices.append(idx)
    logger.info(f"Classifying {len(to_process):,} examples with from_health_professional == False")

    llm_results = []
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    for i in tqdm.tqdm(range(0, len(to_process), batch_size), desc="LLM Classification Batches", ncols=0, total=total_batches):
        batch = to_process[i:i + batch_size]

        # Tokenize batch
        formatted_input = []
        for item in batch:
            if not item["question"].startswith("user:"):
                conv = "user: " + item["question"]
            else:
                conv = item["question"]
            task_prompt = HCP_CLASSIFICATION_PROMPT_PREFIX.format(conversation=conv)
            messages = [
                {"role": "user", "content": task_prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_input.append(formatted_prompt)

        # Tokenize inputs
        tokenized_inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).to(device)

        # Generate output
        with torch.inference_mode():
            outputs = model.generate(
                **tokenized_inputs,
                logits_processor=logits_processor,
                max_new_tokens=1,
                do_sample=False
            ).detach().cpu()

        # Decode output
        batch_results = []
        for input_ids, output_ids in zip(tokenized_inputs.input_ids, outputs):
            # Decode the generated part (excluding the input prompt tokens)
            generated_ids = output_ids[len(input_ids):]
            result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            batch_results.append(result_text.strip())
        llm_results.extend(batch_results)

    # 3. Update dataset "from_health_professional"
    # Map results back to original dataset
    for idx, result in zip(to_process_indices, llm_results):
        result = True if result == "True" else False
        dataset[idx]["from_health_professional"] = result
    logger.info("LLM classification complete.")
    return dataset

###################
# Main
###################
def parse_args():
    parser = argparse.ArgumentParser(description="Process medical QA datasets and optionally run LLM classification.")
    parser.add_argument(
        "--run-llm-classification",
        action="store_true",
        help="Run LLM classification on the filtered dataset."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",  # A small, general-purpose LLM for a default
        help="HuggingFace model name for classification (e.g., 'google/flan-t5-small')."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for LLM inference"
    )
    return parser.parse_args()


def main():
    """Download, preprocess, and optionally classify medical QA datasets."""
    # 1. Define command-line arguments
    args = parse_args()

    # Use the path from the original script
    full_path = f"{DATASET_PATH}/HealthBench/2025-05-07-06-14-12_oss_eval.jsonl"

    # 2. Save HealthBench augmented dataset (full processing)
    logger.info("Starting full dataset processing...")
    d_full = process_healthbench_dataset(path=full_path)
    aug_path = f"{os.environ['PROJECT_DIR']}/data/HealthBench/healthbench_dataset_augmented.jsonl"
    with jsonlines.open(aug_path, "w") as writer:
        writer.write_all(d_full)
    logger.info(f"Saved {len(d_full):,} examples to {aug_path}")

    # 3. Run LLM Classification if requested
    if args.run_llm_classification:
        d_aug_llm = llm_classify_hcp_status(
            dataset=list(d_full),
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        # Update the output path to reflect the new data and classification task
        aug_path = f"{os.environ['PROJECT_DIR']}/data/HealthBench/healthbench_dataset_augmented.jsonl"
        with jsonlines.open(aug_path, "w") as writer:
            writer.write_all(d_aug_llm)
        logger.info(f"Saved {len(d_aug_llm):,} examples to {aug_path}")

    # 3. Apply Filters
    dataset_filter = HealthBenchDataset(
        dataset_path=aug_path,
        include_multiturn=False,
        require_answer=True,
        skip_health_professional=True,
        english_only=True,
        require_completeness_criteria=True,
        skip_negative_criteria=True,
    )
    st_path = f"{os.environ['PROJECT_DIR']}/data/HealthBench/healthbench_dataset_filtered.jsonl"
    with jsonlines.open(st_path, "w") as writer:
        writer.write_all(dataset_filter)
    logger.info(f"Saved {len(dataset_filter):,} examples to {st_path}")


if __name__ == "__main__":
    main()

