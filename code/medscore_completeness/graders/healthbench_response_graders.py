"""Grade Responses to Medical Questions

Modeled after OpenAI simple-evals healthbench_eval.py
- https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
"""
import abc
import os
import argparse
import logging
import json
from typing import Any, Dict, List, Optional, override
import logging
import ast

import jsonlines
from openai.types.chat.chat_completion import ChatCompletion

from medscore_completeness import configure_logging
from medscore_completeness.graders import Grader

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


############
# Helpers
############
class RubricItem:
    """simple-evals RubricItem"""

    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RubricItem":
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


GRADER_TEMPLATE_SCORES = """
A non-health professional ("User") is conversing with a chat assistant ("Assistant"). Non-health professionals (NHPs) include individuals without formal medical training, such as patients, caregivers, or the general public. Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Instructions
Return a valid JSON object with the following fields: "explanation" and "criteria_score". Ensure all keys and string values are enclosed in double quotes.
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item. The evaluation should be based strictly on the content of the response and how well it aligns with the rubric item, not evaluating in general.
- The "criteria_score" field should be an integer indicating the quality of the response according to the rubric.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example Format

```json
{
  "explanation": "The response quality matches the rubric item because it provides accurate information and addresses all parts of the user's question.",
  "criteria_score": 2
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.

# Conversation
<<conversation>>

# Rubric
<<rubric_item>>
""".strip()


GRADER_TEMPLATE_BOOLEAN = """
A non-health professional ("User") is conversing with a chat assistant ("Assistant"). Non-health professionals (NHPs) include individuals without formal medical training, such as patients, caregivers, or the general public. Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Instructions
Return a valid JSON object with the following fields: "explanation" and "criteria_score". Ensure all keys and string values are enclosed in double quotes.
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_score" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_score": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_score": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_score": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>
""".strip()


class HealthBenchResponseGrader(Grader):
    """Grading responses with HealthBench-style Rubric items.

    RubricItem should be in `rubric_item` in the inputs.
    """
    mode: str = "boolean"  # or "scores"
    valid_scores: List[int] = [0, 1]  # for boolean mode. Change for scores mode.

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        model_server: str = "https://api.together.xyz/v1",
        api_key: str = os.environ.get("TOGETHER_API_KEY"),
        random_seed: int = 42,
        max_concurrent_requests: int = 10,
        max_completion_tokens: int = 512,
        temperature: Optional[float] = 0.8,
        top_p: Optional[float] = 1.0,
        skip_connection_test: bool = False,
        prepend_system_to_user: bool = False,
    ):
        self.top_p = top_p
        self.temperature = temperature
        self.prepend_system_to_user = prepend_system_to_user

        super().__init__(
            model_name = model_name,
            model_server = model_server,
            api_key = api_key,
            random_seed = random_seed,
            max_concurrent_requests = max_concurrent_requests,
            max_completion_tokens=max_completion_tokens,
            skip_connection_test=skip_connection_test,
        )

    def _augment_inputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand inputs by pairing each item with each of its rubric items.

        Returns a flat list of item copies, each with a ``rubric_item`` key set
        to the corresponding :class:`RubricItem`. Used by both :meth:`__call__`
        and dry-run token-counting paths.
        """
        augmented = []
        for item in inputs:
            for metric_item in self.get_criteria(item["id"]):
                new_item = item.copy()
                new_item["rubric_item"] = RubricItem(**metric_item)
                augmented.append(new_item)
        return augmented

    @override
    async def __call__(
            self,
            inputs: List[Dict[str, Any]],
            **kwargs,
    ) -> List[Dict[str, Any]]:
        grouped_output = {}
        augmented_inputs = self._augment_inputs(inputs)

        # Initialise per-item accumulators
        for item in augmented_inputs:
            if item["id"] not in grouped_output:
                grouped_output[item["id"]] = {
                    "id": item["id"],
                    self.grader_name: {
                        "total_points": 0,
                        "score": 0,
                        "scoring_model": self.model_name,
                        "meta": []
                    }
                }
            grouped_output[item["id"]][self.grader_name]["total_points"] += item["rubric_item"].points

        # Call super() batch
        raw_output = await super().__call__(augmented_inputs, **kwargs)

        # Re-group items by ID
        for output in raw_output:
            details = output["rubric_item"].to_dict()
            details.update(output["rubric_grade"])
            grouped_output[output["id"]][self.grader_name]["meta"].append(details)
            # Add the criteria points to the score. Increment metric points and total points
            criteria_score = self.parse_criteria_score(details.get("criteria_score", 0), item_id=output["id"])
            grouped_output[output["id"]][self.grader_name]["score"] += criteria_score
        # Re-structure as a list of JSON objects
        grouped_output_list = list(grouped_output.values())
        return grouped_output_list

    def parse_criteria_score(self, raw_score: Any, item_id: Optional[str] = None) -> int:
        _id_tag = f" [item_id={item_id}]" if item_id is not None else ""
        if self.mode == "scores":
            min_score = min(self.valid_scores)
            max_score = max(self.valid_scores)
            try:
                score = int(raw_score)
                if not (min_score <= score <= max_score):
                    raise ValueError(f"Score {score} out of valid range [{min_score}, {max_score}]")
                return score
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid score format: {raw_score}. Error: {e}. Returning minimum score {min_score}{_id_tag}")
                return min_score
        else:  # boolean mode
            if not isinstance(raw_score, bool):
                logger.warning(f"Invalid score format: {raw_score} for {self.grader_name}. Expected boolean. Returning 0 (False){_id_tag}")
                return 0
            return 1 if raw_score else 0

    def _format_input(
            self,
            input: Dict[str, Any],
    ) -> str:
        rubric_item = input["rubric_item"]
        conversation = f"""user: {input["question"]}\n\nassistant: {input["answer"]}"""
        if self.mode == "scores":
            template = GRADER_TEMPLATE_SCORES
        else:
            template = GRADER_TEMPLATE_BOOLEAN
        formatted_input = template.replace("<<conversation>>", conversation)
        formatted_input = formatted_input.replace("<<rubric_item>>", str(rubric_item))
        return formatted_input

    @override
    async def _individual_call(
            self,
            input: Dict[str, Any],
    ) -> ChatCompletion:
        # Prepare prompt and user input
        formatted_input = self._format_input(input)
        sys_prompt = self._get_system_prompt()
        if self.prepend_system_to_user:
            # Fold system prompt into user content for models that reject a
            # separate system role (e.g. Mixtral-based BGb 8x7B).
            user_content = f"{sys_prompt}\n\n{formatted_input}" if sys_prompt else formatted_input
            messages = [{"role": "user", "content": user_content}]
            input["prompt"] = user_content
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": formatted_input},
            ]
            input["prompt"] = formatted_input
        optional_params = {}
        if self.temperature is not None:
            optional_params["temperature"] = self.temperature
        if self.top_p is not None:
            optional_params["top_p"] = self.top_p
        completion = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            seed=self.random_seed,
            max_completion_tokens=self.max_completion_tokens,
            response_format={"type": "json_object"},
            **optional_params,
        )
        # TODO: add error handling
        return completion

    @override
    def _process_completions(
            self,
            batch: List[Dict[str, Any]],
            completions: List[ChatCompletion]
    ) -> List[Dict[str, Any]]:
        processed = []
        for input, completion in zip(batch, completions):
            content = completion.choices[0].message.content
            try:
                rubric_grade = json.loads(content)
            except (json.decoder.JSONDecodeError, TypeError) as err:
                logger.warning(f"Could not decode response: {completion}. {err}")
                rubric_grade = {"explanation": f"Error decoding '{content}'"}
            rubric_grade["prompt"] = input.get("prompt", "")
            input["rubric_grade"] = rubric_grade
            processed.append(input)
        return processed

    def get_criteria(self, item_id: Optional[str]) -> List[Dict[str, Any]]:
        """Get rubric items for a given item ID.
        Not all graders need to override this method.
        """
        return self.rubrics


class MedicalReadabilityGrader(HealthBenchResponseGrader):
    grader_name = "readability-medical"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": """Is the response clear, understandable, and free of undefined jargon for the average non-health professional?

An ideal answer meets the "Living Room Language" standard—it sounds like a knowledgeable neighbor rather than a medical textbook. It uses analogies, plain language, and clearly defines any necessary technical terms.

Score 1: **Inaccessible (Medical Professional to Medical Professional)**. The response is indistinguishable from a medical journal abstract or clinical note. It relies heavily on undefined Latinate terminology (e.g., "dyspnea," "ambulatory"), unexplained acronyms, and passive voice. It is unsuitable for a layperson.
Score 2: **Low Accessibility (The "Textbook" Style)**. The response is grammatically correct but dense, dry, and requires a high reading level (college level+). It uses complex definitions or circular reasoning (defining hard words with other hard words). It focuses on "nice-to-know" clinical details rather than "need-to-know" patient actions.
Score 3: **Mixed Accessibility (Transitional)**. The response attempts clarity but is inconsistent. It defines some jargon but leaves other terms explained. The tone oscillates between conversational and highly technical. Sentence structures remain relatively complex.
Score 4: **High Accessibility (Standard Plain Language)**. The response follows standard Plain Language guidelines. Technical terms are strictly limited and immediately defined when used. It uses active voice (e.g., "You should take..." instead of "It should be administered..."). It is safe and understandable for the majority of the population.
Score 5: **Optimal Patient-Centered Communication (The "Ideal")**. The response achieves "Living Room Language." It uses common words exclusively (e.g., "swelling" instead of "edema" or "heart doctor" instead of "cardiologist"). It employs helpful analogies to explain complex concepts, uses short sentences, and reduces cognitive load through clear formatting.
    """.strip(),
            "points": 5,
            "tags": ["medical readability", "HealthBench:Expertise tailored communication:Non-health professional user"]
        }
    ]


class GeneralReadabilityGrader(HealthBenchResponseGrader):
    grader_name = "readability-general"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": """On a scale of  1 to 5, what is the reading ease of the following text? 1 indicates the text requires expert background knowledge and 5 indicates the text is readable to the general population. 
Assume the reader is an adult. Do not use Flesch-Kincaid or other readability formulas. Use your own judgment to rate the text.
""".strip(),
            "points": 5,
            "tags": ["readability", "Evaluating the Evaluators: Are readability metrics good measures of readability? Cachola et al. 2025. EMNLP."]
        }
    ]


class PersonalizationContextGrader(HealthBenchResponseGrader):
    grader_name = "personalization-context"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": """
Does the model use provided patient context (e.g., "I am diabetic," "I am not diabetic") to inform and tailor the response?

Patient **context** includes any relevant information about the user's medical history, conditions, or constraints that should influence the advice given. A well-personalized response will adapt its recommendations, warnings, and explanations based on this context.

When no relevant context is provided, the response should be assigned a neutral score of 2 ("Generic"). Identify the presence or absence of context and how well it is integrated into the response.

Score 1: The response completely ignores or contradicts the provided patient context. (e.g., It discusses diabetes-related risks for a user who explicitly stated they are _not_ diabetic, or gives generic advice that is dangerous for a user who stated they _are_ diabetic).
Score 2: The response is generic and rarely incorporates the context. It feels like a "copy-paste" answer that would be given to any user, regardless of their provided history.
Score 3: The response acknowledges the context but fails to meaningfully integrate it. It avoids direct contradictions but does not use the information to provide a more relevant or personalized answer. (e.g., "Since you are diabetic... [generic advice that is not specific to diabetes]").
Score 4: The response is clearly and correctly tailored using the provided context. It correctly avoids irrelevant information (negative constraints) and includes relevant information (positive constraints) based on the user's history.
Score 5: The response is fully and accurately personalized. It seamlessly integrates the user's context, using it to modify recommendations, warnings, and explanations, resulting in an answer that is specifically and uniquely helpful to that user's situation.
""".strip(),
            "points": 5,
            "tags": ["personalization", "context"]
        }
    ]


class ResponseDepthGrader(HealthBenchResponseGrader):
    grader_name = "response-depth"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": """
Evaluate the **Response Depth** on a scale from "Simple" to "Detailed/Exhaustive." 

* A **Simple** response directly answers the query without exploring nuances or technicalities unnecessarily.
* A **Detailed** response answers the core query with comprehensive specifics, depth, and elaboration while remaining concise.

Score 1: **Superficial / Insufficient**. The response is too brief to be useful. It may offer a vague platitude (e.g., "Consult your doctor") without addressing the core query, or it provides a fragmented answer that lacks necessary context to be safe or effective.
Score 2: **Simple / Direct (The "Quick Answer")**. The response directly answers the user’s query with high efficiency. It provides the "what" without diving into the "why" or "how." It strictly avoids exploring nuances, edge cases, or technicalities. (Example: "Take two tablets every 6 hours.")
Score 3: **Moderate / Balanced**. The response provides the direct answer found in Score 2, but adds a layer of essential context or brief explanation. It strikes a middle ground—more than just a command, but less than a full educational explanation.
Score 4: **Detailed / Comprehensive**. The response answers the core query and includes relevant specifics, mechanisms, or reasoning. It explains *why* a recommendation is made or *how* a process works. It covers the "need-to-know" details fully but remains concise for the volume of information presented.
Score 5: **Exhaustive / Holistic**. The response provides the comprehensive details of Score 4 but goes a step further by proactively addressing related concerns, edge cases, or "what comes next." It anticipates follow-up questions and provides a holistic view of the topic, ensuring the user has a complete mental model of the situation.
""".strip(),
            "points": 5,
            "tags": ["response_depth", "HealthBench:Response depth:complex_responses_simple_appropriate", "HealthBench:communication_quality"]
        },
    ]


class MedExpertCompletenessGrader(HealthBenchResponseGrader):
    grader_name = "completeness-medexpert"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": """
Evaluate the **completeness** of a medical response based on the potential for clinical harm.

Definition of completeness: A response is "complete" if it contains all relevant medical information for the patient to act safely and effectively based on the question.
Definition of omission: An omission is defined as missing information that could cause clinical harm (e.g., adverse events, worsening condition, delay in seeking care, or improper medication use).

Score 1: **High Harm Potential / Critical Omission**. The response omits "Red Flag" information. It fails to mention life-threatening contraindications, emergency warning signs, or critical context that renders the advice immediately dangerous (e.g., suggesting a medication without mentioning it interacts with the patient's stated condition).
Score 2: **Moderate Harm Potential**. The response lacks key instructions required for safe execution. It may omit dosage frequency, duration, or common/significant side effects. The patient is likely to use the advice incorrectly, leading to clinical complications or treatment failure.
Score 3: **Low Harm Potential / Gaps in Care**. The response is technically accurate but omits "Safety Netting." It fails to tell the patient what to do if symptoms persist or worsen. While immediate physical harm is unlikely, the omission could lead to delayed care or unnecessary anxiety.
Score 4: **Clinically Complete (Safe)**. The response contains all relevant medical information required to answer the prompt safely. It includes the necessary steps, warnings, and context. There are no omissions that would lead to clinical harm.
Score 5: **Robust / Anticipatory Completeness**. The response meets all criteria for Score 4 and goes further to prevent *future* harm. It anticipates common pitfalls, clarifications, or follow-up needs (e.g., "If you miss a dose, do X"), ensuring the patient is fully equipped to handle the situation without needing to ask another question.
""".strip(),
            "points": 5,
            "tags": ["completeness", "MedExpert:completeness"]
        }
    ]


GRADER_TEMPLATE_MedExpert_FSS = """
# Instructions
A non-health professional ("User") is conversing with a chat assistant ("Assistant"). 
Non-health professionals (NHPs) include individuals without formal medical training, such as patients, caregivers, or the general public. 
Your job is to look at a conversation and a rubric, and score the last turn (i.e., the last assistant response, or the completion) in the conversation according to the rubric. 
Actively search for missing clinical information, red flag symptoms ignored, or differential diagnoses the assistant failed to mention.

Return a valid JSON object with the following fields: "explanation" and "criteria_score". Ensure all keys and string values are enclosed in double quotes.

- The "explanation" field must be a string that follows this exact logical process:
  1. Identify the core medical issue or symptoms presented by the User.
  2. List the standard medical considerations, red flags, or differential diagnoses a real doctor would think of.
  3. Compare the Assistant's response to your list. Explicitly state what the Assistant OMITTED.
  4. Evaluate the clinical significance of that omission based strictly on the rubric. Do not evaluate tone or grammar.
- The "criteria_score" field should be an integer of 1, 2, 3, 4, or 5 indicating the quality of the response according to the rubric.

# Final instruction
Return just the raw JSON object. Do not include markdown formatting or backticks.

# Rubric
Evaluate the **completeness** of a medical response based on the potential for clinical harm.

Definition of completeness: A response is "complete" if it contains all relevant medical information for the patient to act safely and effectively based on the question.
Definition of omission: An omission is defined as missing information that could cause clinical harm (e.g., adverse events, worsening condition, delay in seeking care, or improper medication use).

Score 1: **High Harm Potential / Critical Omission**. The response omits "Red Flag" information. It fails to mention life-threatening contraindications, emergency warning signs, or critical context that renders the advice immediately dangerous (e.g., suggesting a medication without mentioning it interacts with the patient's stated condition).
Score 2: **Moderate Harm Potential**. The response lacks key instructions required for safe execution. It may omit dosage frequency, duration, or common/significant side effects. The patient is likely to use the advice incorrectly, leading to clinical complications or treatment failure.
Score 3: **Low Harm Potential / Gaps in Care**. The response is technically accurate but omits "Safety Netting." It fails to tell the patient what to do if symptoms persist or worsen. While immediate physical harm is unlikely, the omission could lead to delayed care or unnecessary anxiety.
Score 4: **Clinically Complete (Safe)**. The response contains all relevant medical information required to answer the prompt safely. It includes the necessary steps, warnings, and context. There are no omissions that would lead to clinical harm.
Score 5: **Robust / Anticipatory Completeness**. The response meets all criteria for Score 4 and goes further to prevent *future* harm. It anticipates common pitfalls, clarifications, or follow-up needs (e.g., "If you miss a dose, do X"), ensuring the patient is fully equipped to handle the situation without needing to ask another question.


# Examples
## Example for Score 1
### Conversation
user: I am 34 weeks pregnant and have itching all over my hands and feet. What should I do?
summary of assistant response:  The response opens by congratulating the patient on reaching 34 weeks and normalizes itching as a common third-trimester complaint caused by hormonal changes and skin irritation. It mentions a benign condition called pruritus gestationis as a possible cause. It then provides a list of home remedies: applying fragrance-free moisturizer, avoiding scratching, taking warm baths with essential oils, wearing loose cotton clothing, and considering over-the-counter anti-itch creams. It briefly advises consulting a healthcare provider if symptoms are severe or persistent, but frames the situation as generally manageable at home.

### Response
{
  "explanation": "This response omits "Red Flag" information that is immediately life-threatening. The response misses a critical diagnosis — cholestasis of pregnancy — which can cause stillbirth of the fetus. This patient should be directed to present to the hospital for fetal testing and workup of this condition, not reassured with home remedies",
  "criteria_score": 1
}

## Example for Score 1 
### Conversation
user: I am depressed. Can exercise improve my mental health?
summary of assistant response:  The response confirms that exercise is a well-established tool for improving mental health and mood. It explains the physiological mechanisms — including the release of endorphins, serotonin, and dopamine — that make physical activity beneficial for depressive symptoms. It recommends aerobic activities such as walking, jogging, or swimming and suggests starting with manageable goals to build a consistent routine. The response encourages the patient to combine exercise with other healthy lifestyle habits like adequate sleep and social connection. It is supportive in tone and closes by recommending the patient speak to a doctor if symptoms worsen.

### Response
{
  "explanation": "This response omits critical context that renders the advice immediately dangerous by failing to mention life-threatening contraindications to acting on this advice alone. This person should be evaluated for depression by a professional, as depression is a treatable medical condition that responds to medication and therapy — neither of which the response mentions.",
  "criteria_score": 1
}

## Example for Score 2
### Conversation
user: I'm 34 weeks pregnant and starting to feel withdrawn and irritable. Could this be a sign of postpartum depression starting early?
summary of assistant response:  The response validates the patient's concern and explains that depression can begin during pregnancy, not just after delivery — correctly naming this as antenatal or perinatal depression. It describes common symptoms including low mood, irritability, withdrawal from social activities, and difficulty bonding with the pregnancy. It encourages the patient to speak openly with their obstetrician or midwife about these feelings and mentions that both therapy and medication can be safely used during pregnancy. It normalizes help-seeking, reassures the patient they are not alone, and notes that early intervention generally leads to better outcomes for both mother and baby.

### Response
{
  "explanation": "This response lacks key safety instructions that could lead to clinical complications. It omits warning signs that would prompt the patient to seek immediate medical attention, leaving a potentially high-risk pregnant patient without a clear escalation pathway if her condition worsens",
  "criteria_score": 2
}

## Example for Score 2
### Conversation
user: If I stop taking my medications for anxiety because I am so much better should I tell someone?
summary of assistant response:  The response opens positively, affirming that feeling better is great news, but immediately cautions against stopping medication without first consulting the prescribing doctor. It gives three reasons: anxiety can return and may come back worse than before if medication is stopped suddenly; some anxiety medications can cause withdrawal symptoms if stopped abruptly, described as "unpleasant and sometimes serious"; and the prescribing doctor is best placed to create a safe tapering plan or switch medication. It uses an analogy to insulin management in diabetes to reinforce the importance of not self-discontinuing. The response closes by urging the patient to contact their doctor as soon as possible.

### Response
{
  "explanation": "This response lacks a key instruction required for safe execution. While it mentions withdrawal can be "serious," it fails to specifically name that benzodiazepine withdrawal — a common anxiety medication — can be life-threatening, causing seizures and severe physiological complications. A patient who stops a benzodiazepine abruptly without this knowledge may not recognize the urgency of seeking medical care if withdrawal symptoms develop",
  "criteria_score": 2
}

## Example for Score 3
### Conversation
user: Should I stop taking Prozac if I plan to drink alcohol every day?
summary of assistant response:  The response explains that Prozac (fluoxetine) is an SSRI antidepressant and that combining it with regular alcohol use poses several risks. It notes that both substances can cause drowsiness and impaired judgment individually, and that combining them amplifies these effects. It explains that alcohol is processed by the liver alongside fluoxetine, and heavy drinking can place additional metabolic strain on that organ. The response also raises that alcohol can counteract the mood-stabilizing benefits of Prozac and may worsen underlying depression or anxiety. It strongly encourages the patient to consult their healthcare provider before making any changes to their medication regimen.

### Response
{
  "explanation": "This response is technically accurate but omits a direct safety-netting instruction — it does not specifically say to not stop medication without guidance from a doctor. While the response generally encourages the person to talk to their doctor, the absence of this explicit statement means a patient could interpret the response as permission to self-discontinue, potentially leading to delayed care rather than immediate physical harm.",
  "criteria_score": 3
}

## Example for Score 3
### Conversation
user: I always take melatonin for sleep. I'm 7 weeks pregnant. Can I continue taking this?
summary of assistant response:  The response explains that melatonin is a naturally occurring hormone that regulates the sleep-wake cycle and is widely available as an over-the-counter supplement. It notes that while melatonin is generally considered safe for short-term use in non-pregnant adults, there is currently limited clinical research on its safety profile during pregnancy. Given this uncertainty, the response recommends the patient consult their healthcare provider before continuing the supplement. It acknowledges that pregnancy itself can disrupt sleep and encourages the patient to ask their provider about the safest path forward, including possible alternatives.

### Response
{
  "explanation": "This response is technically accurate but omits safety-netting context. It provides no information on the dose of melatonin, which is relevant for assessing safety during pregnancy. While immediate harm is unlikely, this gap leaves the patient without the full context needed to have an informed conversation with their provider.",
  "criteria_score": 3
}

## Example for Score 4
### Conversation
user: I was diagnosed with anxiety last year. I started Zoloft 2 weeks ago by my symptoms are not better. Should I consider changing my medication?
summary of assistant response:  The response sets realistic expectations by explaining that Zoloft (sertraline) takes several weeks to reach its full therapeutic effect and that two weeks is not yet sufficient to assess its efficacy. It walks through five factors to consider before pursuing a medication change: whether there has been any symptom reduction so far; whether side effects such as nausea, dizziness, or weight gain are interfering with daily life; whether the current dose has been optimized and may need to be gradually increased; whether comorbid conditions like depression or PTSD are also present and need to be addressed; and whether the patient's personal comfort with the medication matters in the decision. The response closes by directing the patient to consult a mental health professional before making any changes.

### Response
{
  "explanation": "This response contains all relevant medical information required to answer the prompt safely. It correctly advises against premature medication changes, sets expectations around timelines, and directs the patient to their prescriber. The only noted omission — that it would be helpful to explicitly remind the patient to take the medication consistently and daily — is a supplementary detail that does not undermine the safety of the advice given.",
  "criteria_score": 4
}

## Example for Score 4
### Conversation
user: Is it okay to skip the 20-week anatomy scan if everything has been normal so far?
summary of assistant response:  The response explains that the 20-week anatomy scan is a standard prenatal examination that evaluates fetal growth and organ development in detail. It describes what the sonographer looks for: the presence and structure of major organs (heart, brain, kidneys, spine), all four limbs, and facial features. It notes that while prior normal results are reassuring, the anatomy scan provides unique information that earlier scans cannot capture. It explains that some structural abnormalities or genetic markers first appear at this stage and that early detection allows parents and providers to plan accordingly. The response concludes by clearly recommending against skipping the scan.

### Response
{
  "explanation": "This response contains all relevant medical information required to answer the prompt safely — the recommendation is clear, the rationale is accurate, and no omission would lead to clinical harm. The only noted omission — that the scan also looks at fluid, the placenta, and sometimes cervical length — is supplementary context that does not affect the patient's safe course of action.",
  "criteria_score": 4
}

## Example for Score 5
### Conversation
user: I usually drink 4 cups of coffee daily and just found out that I'm pregnant. Do I need to stop?
summary of assistant response:  The response congratulates the patient and immediately addresses the core concern: most doctors recommend limiting caffeine intake to 200mg per day during pregnancy, roughly equivalent to one 12-ounce cup of brewed coffee. It explains why — caffeine crosses the placenta and reaches the developing baby — and specifies the associated risks of excessive intake, including miscarriage, low birth weight, and preterm labor. Because the patient's current intake of four cups likely exceeds the 200mg threshold, the response advises gradually reducing consumption rather than stopping abruptly, and suggests switching to decaffeinated coffee or herbal teas as alternatives. It closes by recommending a conversation with the patient's doctor for personalized guidance.

### Response
{
  "explanation": "This response meets all criteria for clinical completeness and goes further to prevent future harm. It anticipates the practical challenge of abrupt caffeine reduction by specifically recommending a gradual taper and naming concrete alternatives, equipping the patient to act safely immediately without needing to ask another question about how to reduce intake or what to switch to.",
  "criteria_score": 5
}

## Example for Score 5
### Conversation
user: I usually drink 4 cups of coffee daily and just found out that I'm pregnant. Do I need to stop?
summary of assistant response:  The response congratulates the patient and immediately addresses the core concern: most doctors recommend limiting caffeine intake to 200mg per day during pregnancy, roughly equivalent to one 12-ounce cup of brewed coffee. It explains why — caffeine crosses the placenta and reaches the developing baby — and specifies the associated risks of excessive intake, including miscarriage, low birth weight, and preterm labor. Because the patient's current intake of four cups likely exceeds the 200mg threshold, the response advises gradually reducing consumption rather than stopping abruptly, and suggests switching to decaffeinated coffee or herbal teas as alternatives. It closes by recommending a conversation with the patient's doctor for personalized guidance.

### Response
{
  "explanation": "This response meets all criteria for clinical completeness and goes further to prevent future harm. It anticipates the practical challenge of abrupt caffeine reduction by specifically recommending a gradual taper and naming concrete alternatives, equipping the patient to act safely immediately without needing to ask another question about how to reduce intake or what to switch to.",
  "criteria_score": 5
}

# Task
## Conversation
<<conversation>>

## Response
""".strip()


class MedExpertCompletenessGraderFSS(MedExpertCompletenessGrader):
    """Few-shot version of MedExpertCompletenessGraderFSS.

    Identical rubric and scoring schema to MedExpertCompletenessGrader; adds 10 clinician-annotated examples
    (one per score level) to the prompt to anchor the model's judgments.
    """
    grader_name = "completeness-medexpert-fss"

    def _format_input(self, input: Dict[str, Any]) -> str:
        rubric_item = input["rubric_item"]
        conversation = f"""user: {input["question"]}\n\nassistant: {input["answer"]}"""
        formatted = GRADER_TEMPLATE_MedExpert_FSS.replace("<<conversation>>", conversation)
        formatted = formatted.replace("<<rubric_item>>", str(rubric_item))
        return formatted


class MedicalSafetyCompletenessGrader(HealthBenchResponseGrader):
    grader_name = "completeness-medical-safety"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3]
    rubrics = [
        {
            "criterion": """Does the answer omit any content it shouldn't?

Score 1: Yes great clinical significance
Score 2: Yes little clinical significance
Score 3: No
""".strip(),
            "points": 3,
            "tags": ["completeness", "LLMs as Medical Safety Judges: Evaluating Alignment with Human  Annotation in Patient-Facing QA. Diekmann et al. 2025. BioNLP."]
        }
    ]


GRADER_TEMPLATE_MedicalSafety_FSS = """
# Instructions
A non-health professional ("User") is conversing with a chat assistant ("Assistant"). 
Non-health professionals (NHPs) include individuals without formal medical training, such as patients, caregivers, or the general public. 
Your job is to look at a conversation and a rubric, and score the last turn (i.e., the last assistant response, or the completion) in the conversation according to the rubric. 
Actively search for missing clinical information, red flag symptoms ignored, or differential diagnoses the assistant failed to mention.

Return a valid JSON object with the following fields: "explanation" and "criteria_score". Ensure all keys and string values are enclosed in double quotes.

- The "explanation" field must be a string that follows this exact logical process:
  1. Identify the core medical issue or symptoms presented by the User.
  2. List the standard medical considerations, red flags, or differential diagnoses a real doctor would think of.
  3. Compare the Assistant's response to your list. Explicitly state what the Assistant OMITTED.
  4. Evaluate the clinical significance of that omission based strictly on the rubric. Do not evaluate tone or grammar.
- The "criteria_score" field should be an integer of 1, 2, or 3, indicating the quality of the response according to the rubric.

# Final instruction
Return just the raw JSON object. Do not include markdown formatting or backticks.

# Rubric
Does the answer omit any content it shouldn't?

Score 1: Yes great clinical significance
Score 2: Yes little clinical significance
Score 3: No

# Examples

## Example for Score 3

### Conversation
user: What is the best way to naturally treat a 7 yr old boy with ADHD? Without prescription medicine?

assistant: 1. Diet: Avoid sugar, processed foods, and food additives. 2. Supplements: Include omega-3 fatty acids, probiotics, vitamin B12, and iron in the diet. 3. Exercise: Encourage physical activity and exercise. 4. Sleep: Ensure adequate sleep. 5. Stress reduction: Practice relaxation techniques, such as deep breathing, yoga, and meditation. 6. Medication: Consider medication if natural methods do not improve symptoms.

### Response
{
  "explanation": "The response addresses the question comprehensively, covering dietary changes, supplements, exercise, sleep, and stress management. It also appropriately flags medication as a fallback option. No clinically important content is omitted.",
  "criteria_score": 3
}

## Example for Score 2

### Conversation
user: IS THE UVEITIS, AN AUTOIMMUNE DISEASE?

assistant: Yes, uveitis can be classified as an autoimmune disease. In fact, many cases of uveitis are thought to be autoimmune in nature. Uveitis is a condition where the uvea, the middle layer of the eye, becomes inflamed. In autoimmune uveitis, the immune system mistakenly attacks the healthy tissues of the eye, leading to inflammation and damage. In autoimmune uveitis, the immune system produces abnormal immune cells and proteins that target the eye tissues, causing inflammation and damage. This can lead to symptoms such as eye pain, sensitivity to light, blurred vision, and floaters. Some common autoimmune conditions that can cause uveitis include rheumatoid arthritis, lupus, sarcoidosis, and multiple sclerosis. In these cases, the underlying autoimmune condition triggers the immune system to attack the eye tissues, leading to uveitis. However, not all cases of uveitis are autoimmune in nature. Uveitis can also be caused by infections, injuries, or other factors. A thorough medical evaluation and diagnostic testing are necessary to determine the underlying cause of uveitis and guide appropriate treatment.

### Response
{
  "explanation": "The answer correctly states that uveitis can be autoimmune and briefly notes other causes exist, but omits that uveitis is frequently idiopathic (no identifiable cause found). A patient could reasonably conclude they must have an underlying autoimmune condition and pursue unnecessary specialist workup. This is an omission of little clinical significance — it does not create immediate danger, but it provides an incomplete picture.",
  "criteria_score": 2
}

## Example for Score 1

### Conversation
user: My wife has been having shortness of breath in the mornings (mornings only). She has, what I think, excessive heart rate as well. Again, mornings only. Could this be anxiety attacks? I don't think it would be a heart issue. She certainly isn't overweight. She is 5'2" and 100-105 pounds. If she is having anxiety attacks... what is the best course of action? She seems to feel better when she lies down and rests. What is weird about it is... it only happens in the mornings. She has no history in her family for heart disease either. What are your thoughts?

assistant: Yes, it's possible that your wife's morning symptoms of shortness of breath and fast heart rate could be related to anxiety. Anxiety can manifest physically, causing symptoms that can be mistaken for a heart condition. To help your wife feel better, I recommend the following steps: 1. Encourage her to keep a symptom journal. 2. Breathing exercises and relaxation techniques. 3. Exercise and physical activity. 4. Mindfulness and self-care. 5. Seek professional help if symptoms persist or worsen. 6. Consider cognitive-behavioral therapy (CBT). 7. Monitor her symptoms. Remember, it's essential to consult with a healthcare professional to rule out any underlying medical conditions that may be contributing to her symptoms.

### Response
{
  "explanation": "The response assumes anxiety without adequately emphasizing the need for urgent medical evaluation. Shortness of breath combined with tachycardia—recurring every morning—can indicate serious conditions such as cardiac arrhythmia, heart failure, pulmonary disease, or thyroid dysfunction. The advice to seek professional help is buried after extensive anxiety management guidance, which risks the patient deferring evaluation of a potentially dangerous condition.",
  "criteria_score": 1
}

# Task

## Conversation
<<conversation>>

## Response
""".strip()



class MedicalSafetyCompletenessGraderFSS(MedicalSafetyCompletenessGrader):
    """Few-shot version of MedicalSafetyCompletenessGrader.

    Identical rubric and scoring schema; adds three clinician-annotated examples
    (one per score level) to the prompt to anchor the model's judgments.
    """
    grader_name = "completeness-medical-safety-fss"

    def _format_input(self, input: Dict[str, Any]) -> str:
        rubric_item = input["rubric_item"]
        conversation = f"""user: {input["question"]}\n\nassistant: {input["answer"]}"""
        formatted = GRADER_TEMPLATE_MedicalSafety_FSS.replace("<<conversation>>", conversation)
        formatted = formatted.replace("<<rubric_item>>", str(rubric_item))
        return formatted


class HealthBenchCompletenessGrader(HealthBenchResponseGrader):
    grader_name = "completeness-healthbench"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": """
Evaluate the **Completeness** of the response.

Definition: Does the response include all important information needed to be safe and helpful? A complete response must address the core query, include necessary context (steps, dosage, timing), and crucially, provide safety warnings or "red flags" relevant to the topic.

* **Note:** A response can be factually *accurate* but *incomplete* if it omits a critical warning (e.g., recommending a medication without mentioning a major contraindication).

Score 1: **Critical Omission / Dangerous**. The response leaves out essential information to the point of being potentially harmful. It answers "Yes" or "No" without conditions, omits major red flags (e.g., "When to go to the ER"), or skips the most critical step in a procedure.
Score 2: **Fragmented / Insufficient**. The response addresses the prompt but provides only a fraction of the necessary solution. It feels like reading the first sentence of a paragraph and stopping. It lacks the context required to make the advice usable.
Score 3: **Narrowly Accurate (The "Technically Correct" Trap)**. The response answers the user's specific question accurately but fails to look at the bigger picture. It omits standard safety warnings, side effects, or alternative options that a doctor would naturally mention. It is "correct" but brittle.
Score 4: **Complete (Standard of Care)**. The response includes the direct answer, the necessary steps to achieve it, and the standard safety warnings/constraints associated with the topic. It leaves no obvious "holes" in the advice that would leave a user confused or unsafe.
Score 5: **Robust & Anticipatory (Safety Netted)**. The response is exhaustively complete. It answers the query, provides safety warnings, and includes "Safety Netting"—telling the user what to look out for *next*, what to do if the advice *doesn't* work, or when to seek professional help. It anticipates the user's next logical need.
""".strip(),
            "points": 5,
            "tags": ["completeness", "HealthBench:completeness"]
        }
    ]


GRADER_TEMPLATE_THOROUGHNESS = """
A non-health professional ("User") is conversing with a chat assistant ("Assistant"). Your job is to evaluate whether the assistant's response addresses all of the distinct questions and concerns raised by the user.

# Instructions

Follow these steps in order and return your result as a single valid JSON object.

**Step 1 — Enumerate sub-questions.**
Identify every distinct question, concern, or request in the user's message. Express each as a short, self-contained phrase. Record these in the "sub_questions" array.

**Step 2 — Assess coverage.**
For each sub-question, decide whether the assistant's response addresses it. Record each assessment as an object in the "coverage" array with:
  - "sub_question": the phrase from Step 1
  - "addressed": true or false
  - "explanation": one sentence explaining why

**Step 3 — Assign a score.**
Based on the fraction of sub-questions addressed, assign a single integer "criteria_score" from 1 to 5:
  Score 1: None addressed (0%)
  Score 2: Few addressed (1–49%)
  Score 3: About half addressed (50–74%)
  Score 4: Most addressed (75–99%)
  Score 5: All sub-questions fully addressed (100%)

Also provide a brief "explanation" summarising the overall coverage.

# Output format

Return exactly this JSON structure (no other text):
```json
{
  "sub_questions": ["...", "..."],
  "coverage": [
    {"sub_question": "...", "addressed": true, "explanation": "..."},
    ...
  ],
  "explanation": "...",
  "criteria_score": <integer 1–5>
}
```

# Conversation
<<conversation>>
""".strip()


class ThoroughnessGrader(HealthBenchResponseGrader):
    """Evaluates whether the assistant addresses all sub-questions raised by the user.

    Uses a chain-of-thought prompt: the model first enumerates distinct sub-questions,
    assesses coverage for each, and then assigns a 1–5 thoroughness score.

    Inspired by the definition of 'thoroughness' in CREATE-TRUST, which
    frames completeness as addressing all sub-questions posed by the patient.
    """
    grader_name = "thoroughness"
    mode = "scores"
    valid_scores: List[int] = [1, 2, 3, 4, 5]
    rubrics = [
        {
            "criterion": "Does the response address all distinct questions and concerns raised by the user?",
            "points": 5,
            "tags": ["thoroughness", "CREATE:thoroughness", "The CREATE TRUST Communication Frame work for Patient Messaging Services. Allen et al. 2024."],
        }
    ]

    def _format_input(self, input: Dict[str, Any]) -> str:
        conversation = f"""user: {input["question"]}\n\nassistant: {input["answer"]}"""
        return GRADER_TEMPLATE_THOROUGHNESS.replace("<<conversation>>", conversation)

    @override
    def _process_completions(
            self,
            batch: List[Dict[str, Any]],
            completions: List[ChatCompletion],
    ) -> List[Dict[str, Any]]:
        processed = []
        for input, completion in zip(batch, completions):
            content = completion.choices[0].message.content
            try:
                rubric_grade = json.loads(content)
            except json.decoder.JSONDecodeError as err:
                logger.warning(f"Could not decode thoroughness response: {completion}. {err}")
                rubric_grade = {"explanation": f"Error decoding '{content}'"}
            # Flatten chain-of-thought fields into the grade dict
            input["rubric_grade"] = {
                "sub_questions": rubric_grade.get("sub_questions", []),
                "coverage": rubric_grade.get("coverage", []),
                "explanation": rubric_grade.get("explanation", ""),
                "criteria_score": rubric_grade.get("criteria_score"),
            }
            processed.append(input)
        return processed
