"""Generate medical question-answering responses using a Hugging Face model"""
import os
import argparse
import logging
from typing import Any, Dict, List, Optional, override
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import jsonlines

from medscore_completeness import configure_logging
from medscore_completeness.utils.medical_qa_dataset import medical_qa_to_class


################
# Helper functions
################
GENERATION_PROMPT = """### Instruction
You are a helpful medical assistant. Please answer the **Patient Question** below. 
Your response must strictly adhere to the following **Guidance Criteria**. 
Ensure your advice covers the relevant mitigation measures and triage recommendations mentioned in the criteria.

### Guidance Criteria
{criteria_text}

### Patient Question
{question_text}

### Response
"""

class ChatDataset(torch.utils.data.Dataset):
    """Dataset wrapper to apply chat templates to questions."""
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]

        # Format the prompt
        question = item.get("question", "")
        criteria = item.get("criteria", [])
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        formatted_question = GENERATION_PROMPT.format(
            criteria_text=criteria_text,
            question_text=question
        )

        # Create the chat structure required by instruction-tuned models
        system_prompt = "You are an AI-powered helpful health information assistant."
        if "medgemma" in self.model_name.lower():
            system_prompt = f"SYSTEM INSTRUCTION: think silently if needed. {system_prompt}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_question}
        ]

        # Apply the chat template to format the input string
        # tokenize=False allows us to handle tokenization/padding in the collator
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "prompt": prompt,
            "original_item": item
        }


class ChatCollator:
    """Collator to tokenize batches of chat prompts."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = [x["prompt"] for x in batch]
        original_items = [x["original_item"] for x in batch]

        # Tokenize inputs with dynamic padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return inputs, original_items


############
# Main
############
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate medical QA responses")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="HealthBench", choices=medical_qa_to_class.keys())
    parser.add_argument("--dataset-path", type=str, default=None, help="Overwrite default dataset path")
    parser.add_argument("--n-examples", type=int, default=-1, help="Number of examples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Generation settings
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--model-name", type=str, default="google/medgemma-27b-text-it", help="Model type")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    # Configure logging
    configure_logging(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        log_file=os.path.join(args.output_dir, "generation.log")
    )
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset for inference
    dataset_class = medical_qa_to_class[args.dataset]
    logger.info(f"Loading dataset {args.dataset} from {args.dataset_path}")
    dataset = dataset_class(
        dataset_path=args.dataset_path,
        n_examples=args.n_examples,
        random_state=args.seed,
    )
    logger.debug(dataset[0])

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = "auto" if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.model_max_length = model.config.max_position_embeddings
    model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Model and tokenizer loaded: {args.model_name}")

    # Create Dataset and DataLoader
    chat_dataset = ChatDataset(dataset, tokenizer)
    collator = ChatCollator(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        chat_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )

    # Batch generate
    output_path = os.path.join(args.output_dir, "generated.jsonl")
    writer = jsonlines.open(output_path, mode="w")
    logger.info(f"Generating responses and saving to {output_path}")
    for input_batch, data_batch in tqdm(dataloader, desc="Generating responses"):
        tokenized_inputs = input_batch.to(device)

        # Generate outputs
        with torch.inference_mode():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,  # This will throw a warning if temperature=0 and do_sample=False
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            ).detach().cpu()

        input_lengths = tokenized_inputs['input_ids'].shape[1]
        for data, o in zip(data_batch, outputs):
            # Extract generated tokens
            generated_tokens = o[input_lengths:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            if "medgemma" in args.model_name.lower():
                # Regex to capture the thought (group 1) and the response (group 2)
                # re.DOTALL ensures the dot (.) matches newlines
                pattern = r"<unused94>thought\n(.*?)<unused95>(.*)"
                match = re.search(pattern, response, re.DOTALL)

                if match:
                    thought = match.group(1).strip()
                    response = match.group(2).strip()
                else:
                    # Fallback if the pattern is not found
                    thought = ""
            else:
                thought = None
            row = data.copy()
            row["generated_response"] = response
            row["model_reasoning"] = thought
            writer.write(row)
    writer.close()


if __name__ == "__main__":
    main()

