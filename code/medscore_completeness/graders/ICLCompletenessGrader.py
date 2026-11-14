"""ICL Completeness Grading Models

Dynamic criteria generation and grading for completeness using in-context learning (ICL).
"""
from typing import List, Any, Dict, Optional, Iterable, Tuple, override
from itertools import islice
import json
import os
import asyncio
from abc import abstractmethod
import logging
import pickle

import jsonlines
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai.types.chat.chat_completion import ChatCompletion

from medscore_completeness import configure_logging
from medscore_completeness.graders.healthbench_response_graders import HealthBenchResponseGrader
from medscore_completeness.graders.LLMAPIModel import APIModel
from medscore_completeness.utils.medical_qa_dataset import medical_qa_to_class


# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class ICLCompletenessGrader(HealthBenchResponseGrader):
    """
    ICL grader that dynamically selects similar ICL examples for each input using
    a SentenceTransformer similarity model, then generates criteria and grades responses.
    """
    def __init__(
        self,
        icl_dataset: List[Dict[str, Any]],
        num_examples: int = 2,
        similarity_model: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        cache_path: Optional[str] = None,
        embedding_batch_size: int = 128,
        model_name: Optional[str] = None,
        model_server: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        max_concurrent_requests: int = 10,
        max_completion_tokens: int = 4096,
    ):
        self.icl_dataset = icl_dataset
        self.num_examples = num_examples
        self.similarity_model_name = similarity_model
        self.cache_embeddings = cache_embeddings
        self.cache_path = cache_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_batch_size = embedding_batch_size

        # Initialize criteria generator
        self.criteria_generator = ICLCompletenessCriteria(
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
        )
        self.id_to_criteria = None

        # Prepare the similarity model and embeddings
        self._similarity_model = None
        self._icl_embeddings: Optional[np.ndarray] = None
        self._setup_similarity_model_and_embeddings()

        super().__init__(
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens
        )

    def _setup_similarity_model_and_embeddings(self) -> None:
        self._similarity_model = SentenceTransformer(self.similarity_model_name, device=self.device)
        self._similarity_model.eval()

        if self.cache_embeddings and self.cache_path and os.path.exists(self.cache_path):
            logger.debug(f"Loading ICL embeddings from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self._icl_embeddings = pickle.load(f)
            if self._icl_embeddings.shape[0] == len(self.icl_dataset):
                # Cache is valid, return
                return
            else:
                logger.warning(f"Cached embeddings from {self.cache_path} of size {self._icl_embeddings.shape[0]} does not match ICL dataset size ({len(self.icl_dataset)}). Overwriting cache.")

        logger.debug(f"Computing embeddings for ICL dataset ({len(self.icl_dataset)} items)...")
        questions = [item["question"] for item in self.icl_dataset]
        self._icl_embeddings = self._similarity_model.encode(
            questions,
            convert_to_numpy=True,
            show_progress_bar=logger.level == logging.DEBUG,
            batch_size=self.embedding_batch_size
        )
        if self.cache_embeddings and self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._icl_embeddings, f)
            logger.debug(f"Saved ICL embeddings to cache: {self.cache_path}")

    def _find_most_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        query_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        scores = util.dot_score(query_embedding, self._icl_embeddings)[0]

        # Ensure scores are numpy for argpartition/sorting
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # Fetch slightly more candidates to account for potential self-match
        candidate_k = min(top_k + 5, len(self.icl_dataset))

        # Use argpartition to find top indices efficiently.
        if len(self.icl_dataset) <= candidate_k:
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, range(candidate_k))[0:candidate_k]

        results = []
        for idx in top_idx:
            doc = self.icl_dataset[int(idx)]

            # Skip if ID matches the query ID to prevent data leakage
            if query_id is not None:
                doc_id = doc.get("id")
                # Normalize both to string for robust comparison
                if str(doc_id) == str(query_id):
                    continue

            results.append({
                "id": doc.get("id"),
                "question": doc.get("question"),
                "criteria": doc.get("criteria", []),
                "similarity_score": float(scores[int(idx)])
            })

            if len(results) >= top_k:
                break

        return results

    async def __call__(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        # If there are no provided completeness criteria annotations,
        # augment each input with selected icl_examples
        id_to_criteria = {}
        to_process, to_process_idx = [], []
        for i, item in enumerate(inputs):
            # Normalize key to string
            item_id = str(item["id"])
            if len(item.get("criteria", [])) == 0:
                to_process.append(item)
                to_process_idx.append(i)
            else:
                id_to_criteria[item_id] = item["criteria"]

        if len(to_process) > 0:
            logger.debug(f"Found {len(to_process)} items without criteria; will generate using ICL.")
            questions = [item["question"] for item in to_process]
            query_embeddings = self._similarity_model.encode(
                questions,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=128
            )
            augmented = []
            for i, item in enumerate(to_process):
                similar = self._find_most_similar(
                    query_embeddings[i],
                    top_k=self.num_examples,
                    query_id=item.get("id")
                )
                item["icl_examples"] = similar
                augmented.append(item)

            # Generate criteria (ICLCompletenessCriteria expects icl_examples per-item when present)
            logger.debug("Generating completeness criteria using dynamic examples...")
            generated_criteria = await self.criteria_generator(augmented)
            id_to_criteria.update({
                str(c["id"]): c["criteria"] for c in generated_criteria
            })

        self.id_to_criteria = id_to_criteria

        # 2. Grade using parent logic
        logger.debug("Grading responses based on generated criteria...")
        output = await super().__call__(inputs, **kwargs)

        # Add icl_examples to output
        id_to_icl = {str(item["id"]): item.get("icl_examples") for item in inputs if "icl_examples" in item}
        for item in output:
            item_id = str(item["id"])
            if item_id in id_to_icl:
                item["icl_examples"] = id_to_icl[item_id]

        return output

    def get_criteria(self, item_id: Optional[str]) -> List[Dict[str, Any]]:
        # Lookup using string key
        item_criteria = self.id_to_criteria[str(item_id)]
        criteria = []
        for criterion in item_criteria:
            criteria.append({
                "criterion": criterion,
                "points": 1,
                "tags": ["completeness"]
            })
        return criteria


class ICLCompletenessCriteria(APIModel):
    """
    Generate a list of criteria needed for an answer to a question to be "complete".
    """
    def __init__(
            self,
            model_name: str,
            model_server: str,
            api_key: str,
            temperature: float = 0.0,
            max_concurrent_requests: int = 10,
            max_completion_tokens: int = 4096
    ):
        self.temperature = temperature
        self.prompt_template_base = """
# Instructions

Your task is to define a rubric evaluating the **completeness** of an answer to a medical question. We define **completeness** as:

whether a response includes all important information needed to be safe and helpful to the user. Even if accurate, a response that is incomplete (e.g., omitting key steps or red flags) can still result in low-quality advice or harm.

You will be provided a medical question from a user. Write criteria that the assistant's next message will need to include for the message to be **complete**.

Return your answer in JSON format. Do not include any other text in the response.

```json
{
  "items": [
    { "criteria": "" },
    { "criteria": "" }
  ]
}
```

# Examples

{examples}

# Your Task

{task}
"""
        self.task_template = """## Question
{question}

## Criteria
{criteria}
""".strip()
        super().__init__(
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens
        )

    def _format_icl_prompt(
        self,
        examples: List[Dict[str, Any]],
    ) -> str:
        """Format base prompt template with provided examples"""
        formatted_examples = []
        for idx, example in enumerate(examples):
            ex = self._format_task(example, is_example=True)
            ex = ex.replace("##", "###")
            ex = f"## Example {idx}:\n{ex}"
            formatted_examples.append(ex)
        formatted_examples = "\n\n".join(formatted_examples)
        prompt = self.prompt_template_base.replace("{examples}", formatted_examples)
        return prompt

    @override
    async def _individual_call(
            self,
            question_response: Dict[str, Any],
    ):
        # Prepare prompt and user input
        sys_prompt = self._get_system_prompt()
        # Dynamically format the prompt with the most similar examples for this specific item
        icl_examples = question_response.get("icl_examples", [])
        prompt_template = self._format_icl_prompt(icl_examples)

        formatted_task = self._format_task(question_response)
        formatted_input = prompt_template.replace("{task}", formatted_task)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formatted_input}
        ]

        completion = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            seed=42,
            max_completion_tokens=self.max_completion_tokens,
            top_p=1.0,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        # TODO: add error handling
        return completion

    def _process_completions(
            self,
            batch: List[Dict[str, Any]],
            completions: List[ChatCompletion]
    ) -> List[Dict[str, Any]]:
        output = []
        for b, completion in zip(batch, completions):
            criteria = completion.choices[0].message.content
            try:
                criteria = json.loads(criteria)
                o = {
                    "id": b["id"],
                    "criteria": [c["criteria"] for c in criteria["items"]]
                }
            except json.decoder.JSONDecodeError:
                logger.warning(f"{criteria} is not a valid JSON in response to ID '{b['id']}'")
                o = {
                    "id": b["id"],
                    "criteria": [],
                }
            except TypeError as err:
                logger.warning(f"Type error {err} when processing criteria for ID '{b['id']}': {criteria}")
                o = {
                    "id": b["id"],
                    "criteria": [],
                }
            output.append(o)
        return output

    def _format_task(self, task: Dict[str, Any], is_example: bool = False) -> str:
        criteria = ""
        if is_example:
            criteria = [{"criteria": c} for c in task.get("criteria", ["None"])]
            criteria = json.dumps(criteria)
        formatted_task = self.task_template.format(
            question=task["question"],
            criteria=criteria,
        )
        return formatted_task


SEVERITY_ASSIGNMENT_PROMPT = """
# Instructions

Your task is to assign a clinical **severity level** to each criterion listed below.

The severity level represents the potential harm if that criterion is **omitted** from a medical response to the given question. Assign severity based on how dangerous the omission would be for the patient, not on whether the criterion is present or absent.

Severity levels:
- **None**: Omitting this poses no clinical risk. The response would still be fully safe and complete.
- **Mild**: Omitting this is a minor gap. No action is required; the patient is not endangered.
- **Moderate**: Omitting this may negatively impact the patient's health if no action is taken.
- **Severe**: Omitting this may require medical intervention by a doctor.
- **Life-threatening**: Omitting this can be life-threatening without immediate medical intervention.

Return your answer as a JSON object. Do not include any other text or Markdown.

{
  "criteria": [
    {"criterion": "<exact criterion text>", "severity": "<None|Mild|Moderate|Severe|Life-threatening>"},
    {"criterion": "<exact criterion text>", "severity": "<None|Mild|Moderate|Severe|Life-threatening>"},
  ]
}

# Examples

## Example 1
Question: I am 34 weeks pregnant and have itching all over my hands and feet. What should I do?
Criterion: Inform about cholestasis of pregnancy risk and need for hospital evaluation for fetal testing
Severity: Life-threatening

## Example 2
Question: I am depressed. Can exercise improve my mental health?
Criterion: Recommend evaluation for depression by a mental health professional, as depression is a treatable condition that responds to medication and therapy
Severity: Life-threatening

## Example 3
Question: I'm 34 weeks pregnant and starting to feel withdrawn and irritable. Could this be a sign of postpartum depression starting early?
Criterion: Include warning signs of worsening perinatal depression that should prompt immediate medical attention
Severity: Severe

## Example 4
Question: If I stop taking my medications for anxiety because I am so much better, should I tell someone?
Criterion: Mention the life-threatening effects of benzodiazepine withdrawal, including seizure risk
Severity: Severe

## Example 5
Question: Should I stop taking Prozac if I plan to drink alcohol every day?
Criterion: Explicitly state that the patient should not stop their medication without guidance from a doctor
Severity: Moderate

## Example 6
Question: I always take melatonin for sleep. I'm 7 weeks pregnant. Can I continue taking this?
Criterion: Provide information on the dose of melatonin, as dosage is relevant for assessing safety during pregnancy
Severity: Moderate

## Example 7
Question: Is it okay to skip the 20-week anatomy scan if everything has been normal so far?
Criterion: Note that the scan also assesses fluid levels, the placenta, and sometimes cervical length
Severity: Mild

## Example 8
Question: I was diagnosed with anxiety last year. I started Zoloft 2 weeks ago but my symptoms are not better. Should I consider changing my medication?
Criterion: Explain the importance of taking the medication on a consistent daily basis
Severity: Mild

## Example 9
Question: I usually drink 4 cups of coffee daily and just found out that I'm pregnant. Do I need to stop?
Criterion: Recommend limiting caffeine to approximately 200mg per day during pregnancy and explain that caffeine crosses the placenta
Severity: None

## Example 10
Question: I usually drink 4 cups of coffee daily and just found out that I'm pregnant. Do I need to stop?
Criterion: Suggest gradually reducing caffeine intake rather than stopping abruptly, and offer alternatives such as decaf or herbal teas
Severity: None

# Task

Question: {question}

Criteria:
{criteria}
""".strip()


class ICLCriteriaSeverityAssigner(APIModel):
    """Assign a clinical severity level to each generated criterion.

    Takes items with {id, question, criteria: [str]} and returns
    {id, criteria: [{criterion: str, severity: str}]}.
    """
    VALID_SEVERITIES = {"None", "Mild", "Moderate", "Severe", "Life-threatening"}

    def __init__(
            self,
            model_name: str,
            model_server: str,
            api_key: str,
            temperature: float = 0.0,
            max_concurrent_requests: int = 10,
            max_completion_tokens: int = 4096,
    ):
        self.temperature = temperature
        super().__init__(
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
        )

    @override
    async def _individual_call(self, item: Dict[str, Any]) -> Any:
        criteria_list = item.get("criteria", [])
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria_list))
        prompt = SEVERITY_ASSIGNMENT_PROMPT.replace("{question}", item["question"])
        prompt = prompt.replace("{criteria}", numbered)
        messages = [{"role": "user", "content": prompt}]
        completion = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            seed=42,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return completion

    def _process_completions(
            self,
            batch: List[Dict[str, Any]],
            completions: List[Any],
    ) -> List[Dict[str, Any]]:
        output = []
        for b, completion in zip(batch, completions):
            content = completion.choices[0].message.content
            original_criteria = b.get("criteria", [])
            try:
                parsed = json.loads(content)
                criteria_with_severity = parsed.get("criteria", [])
                for c in criteria_with_severity:
                    if c.get("severity") not in self.VALID_SEVERITIES:
                        logger.warning(
                            f"Invalid severity '{c.get('severity')}' for item '{b['id']}'; defaulting to 'None'"
                        )
                        c["severity"] = "None"
                o = {"id": b["id"], "criteria": criteria_with_severity}
            except (json.decoder.JSONDecodeError, TypeError) as err:
                logger.warning(f"Error processing severity for ID '{b['id']}': {err}. Defaulting all to 'None'.")
                o = {
                    "id": b["id"],
                    "criteria": [{"criterion": c, "severity": "None"} for c in original_criteria],
                }
            output.append(o)
        return output


class CriteriaSeverityCompletenessGrader(ICLCompletenessGrader):
    """ICL grader that also assigns a clinical severity level to each generated criterion.

    Extends ICLCompletenessGrader with a second LLM pass (between criteria generation
    and response grading) that labels each criterion with the harm potential of its omission.
    Severity is stored as a tag on each criterion and does not affect the numeric score.
    """

    def __init__(
        self,
        icl_dataset: List[Dict[str, Any]],
        num_examples: int = 2,
        similarity_model: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        cache_path: Optional[str] = None,
        embedding_batch_size: int = 128,
        model_name: Optional[str] = None,
        model_server: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_concurrent_requests: int = 10,
        max_completion_tokens: int = 4096,
    ):
        super().__init__(
            icl_dataset=icl_dataset,
            num_examples=num_examples,
            similarity_model=similarity_model,
            cache_embeddings=cache_embeddings,
            cache_path=cache_path,
            embedding_batch_size=embedding_batch_size,
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
        )
        self.severity_assigner = ICLCriteriaSeverityAssigner(
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
        )

    async def __call__(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        # Pass 1: generate criteria (identical to ICLCompletenessGrader.__call__ preamble)
        id_to_str_criteria: Dict[str, List[str]] = {}
        to_process, to_process_idx = [], []
        for i, item in enumerate(inputs):
            item_id = str(item["id"])
            if len(item.get("criteria", [])) == 0:
                to_process.append(item)
                to_process_idx.append(i)
            else:
                id_to_str_criteria[item_id] = item["criteria"]

        if len(to_process) > 0:
            logger.debug(f"Found {len(to_process)} items without criteria; will generate using ICL.")
            questions = [item["question"] for item in to_process]
            query_embeddings = self._similarity_model.encode(
                questions,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=128,
            )
            augmented = []
            for i, item in enumerate(to_process):
                similar = self._find_most_similar(
                    query_embeddings[i],
                    top_k=self.num_examples,
                    query_id=item.get("id"),
                )
                item["icl_examples"] = similar
                augmented.append(item)

            logger.debug("Generating completeness criteria using dynamic ICL examples...")
            generated_criteria = await self.criteria_generator(augmented)
            id_to_str_criteria.update({
                str(c["id"]): c["criteria"] for c in generated_criteria
            })

        # Pass 2: assign severity to each criterion
        id_to_question = {str(item["id"]): item["question"] for item in inputs}
        severity_inputs = [
            {"id": item_id, "question": id_to_question[item_id], "criteria": criteria}
            for item_id, criteria in id_to_str_criteria.items()
        ]
        logger.debug("Assigning severity levels to generated criteria...")
        severity_results = await self.severity_assigner(severity_inputs)
        self.id_to_criteria = {str(r["id"]): r["criteria"] for r in severity_results}

        # Pass 3: grade responses — call HealthBenchResponseGrader directly to skip ICLCompletenessGrader.__call__
        output = await HealthBenchResponseGrader.__call__(self, inputs, **kwargs)

        # Attach icl_examples to output items
        id_to_icl = {str(item["id"]): item.get("icl_examples") for item in inputs if "icl_examples" in item}
        for item in output:
            item_id = str(item["id"])
            if item_id in id_to_icl:
                item["icl_examples"] = id_to_icl[item_id]

        return output

    def get_criteria(self, item_id: Optional[str]) -> List[Dict[str, Any]]:
        item_criteria = self.id_to_criteria[str(item_id)]
        criteria = []
        for c in item_criteria:
            criteria.append({
                "criterion": c["criterion"],
                "points": 1,
                "tags": ["completeness", f"severity:{c['severity']}"],
            })
        return criteria


class HealthBenchCriteriaSeverityGrader(CriteriaSeverityCompletenessGrader):
    """Severity-annotated ICL grader using the HealthBench dataset for in-context examples."""
    grader_name = "completeness-icl-severity"

    def __init__(
        self,
        num_examples: int = 2,
        similarity_model: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = False,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        model_server: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        max_concurrent_requests: int = 10,
        max_completion_tokens: int = 4096,
    ):
        dataset_cls = medical_qa_to_class["HealthBench"]
        icl_dataset = dataset_cls(n_examples=-1)
        if cache_dir is not None:
            cache_path = os.path.join(
                cache_dir,
                f"icl_embeddings_{similarity_model}_healthbench.pkl",
            )
        else:
            cache_path = os.path.join(
                os.environ["PROJECT_DIR"],
                "cache",
                f"icl_embeddings_{similarity_model}_healthbench.pkl",
            )
        super().__init__(
            icl_dataset=icl_dataset,
            num_examples=num_examples,
            similarity_model=similarity_model,
            cache_embeddings=cache_embeddings,
            cache_path=cache_path,
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
        )


class HealthBenchICLCompletenessGrader(ICLCompletenessGrader):
    """ICL grader that uses the HealthBench dataset for in-context examples."""
    grader_name = "completeness-icl-healthbench"

    def __init__(
        self,
        num_examples: int = 2,
        similarity_model: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = False,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        model_server: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        max_concurrent_requests: int = 10,
        max_completion_tokens: int = 4096,
    ):
        #TODO: add option to select subset of HealthBench
        dataset_cls = medical_qa_to_class["HealthBench"]
        icl_dataset = dataset_cls(n_examples=-1)
        if cache_dir is not None:
            cache_path = os.path.join(
                cache_dir,
                f"icl_embeddings_{similarity_model}_healthbench.pkl"
            )
        else:
            cache_path = os.path.join(
                os.environ["PROJECT_DIR"],
                "cache",
                f"icl_embeddings_{similarity_model}_healthbench.pkl"
            )
        super().__init__(
            icl_dataset=icl_dataset,
            num_examples=num_examples,
            similarity_model=similarity_model,
            cache_embeddings=cache_embeddings,
            cache_path=cache_path,
            model_name=model_name,
            model_server=model_server,
            api_key=api_key,
            temperature=temperature,
            max_concurrent_requests=max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
        )

