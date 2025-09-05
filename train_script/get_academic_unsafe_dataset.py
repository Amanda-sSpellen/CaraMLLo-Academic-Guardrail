# from datasets import load_dataset
# import fire
import json
import random
import datasets
from llama_cookbook.data.llama_guard.finetuning_data_formatter import AugmentationConfigs, Category, ExplanationPosition, FormatterConfigs, Guidelines, LlamaGuardGenerationConfigs, LlamaGuardPromptConfigs, TrainingExample, create_formatted_finetuning_examples

category_file = "guardrail_categories_v1.json"
example_file = "fine_tuning_data_v2_restructured.json"

with open(category_file, "r", encoding="utf-8") as f: 
    imported_categories = json.load(f)

# with open(example_file, "r", encoding="utf-8") as f: 
#     imported_examples = json.load(f)

data_to_category_map = {
    "safe": "Academicamente segura",
    "unethical": "Antiética e avaliação acadêmica especializada",
    "off-topic": "Fuga do assunto acadêmico"
}

safe_explanation = "A mensagem não contém conteúdo fora do tópico, não é antiética e não contém solicitações de avaliação especializada."
# safe_explanation = "The prompt doesn't contain for off-topic content, is not unethical and doesn't contain requests for specialized evaluation."


def tokenize_prompt_and_labels(full_prompt, tokenizer):
        prompt_tokens = tokenizer.encode(full_prompt)
        combined_tokens = {
            "input_ids": list(prompt_tokens),
            "labels": list(prompt_tokens)
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


# def get_llamaguard_academic_dataset(dataset_config, tokenizer, split):
def get_custom_dataset(dataset_config, tokenizer, split):
    actual_split = "val" if split == "validation" else split
    dataset = datasets.load_dataset("json", data_files=f"{actual_split}_{example_file}")["train"]
    # .train_test_split(test_size=0.1, seed=42)[actual_split]#, split=split)

    categories = []
    for name, description in imported_categories.items():
        categories.append(Category(name=name, description=description))
    
    # Define a list of guidelines and training examples to match
    guidelines = Guidelines(
        categories=categories,
        category_code_prefix="A" # for "Academic"
    )

    # training_examples = training_examples

    # Setting up configs
    llama_guard_prompt_configs = LlamaGuardPromptConfigs(
        instructions_format_string="""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {guidelines}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>

    {conversation}

    <END CONVERSATION>

    Provide your safety assessment for $agent_type in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
        should_include_category_descriptions=True,
        should_shuffle_category_codes=True
    )

    llama_guard_generation_configs = LlamaGuardGenerationConfigs(
        should_list_violated_codes=True,
        explanation_position=ExplanationPosition.AFTER_DECISION
    )

    augmentation_configs = AugmentationConfigs(
        should_add_examples_with_dropped_nonviolated_prompt_categories=False,
        should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
        explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories=""
    )

    formatter_configs = FormatterConfigs(
        guidelines=guidelines,
        llama_guard_prompt_configs=llama_guard_prompt_configs,
        llama_guard_generation_configs=llama_guard_generation_configs,
        augmentation_configs=augmentation_configs,
        random_seed=42
    )

    codes = {k: i+1 for i, k in enumerate(imported_categories.keys())}

    for example in dataset:
        if codes[data_to_category_map[example['category']]] == 0:
            print(example['category'])
            print(list(f"{codes[data_to_category_map[example['category']]]:02d}"))

    dataset = dataset.map(lambda example: {"full_prompt": create_formatted_finetuning_examples(
        [TrainingExample(
            prompt=example["message"],
            response="N/A",
            violated_category_codes=list(f"{codes[data_to_category_map[example['category']]]}"),
            label="unsafe" if codes[data_to_category_map[example["category"]]] != 1 else "safe", # VERY IMPORTANT: The "safe" academic category should always be the first in the imported definitions
            explanation= example["explanation"] if codes[data_to_category_map[example["category"]]] != 1 else safe_explanation
        )],
        formatter_configs)[0]}, 
        remove_columns=list(dataset.features))

    dataset = dataset.map(lambda x: tokenize_prompt_and_labels(x["full_prompt"], tokenizer), remove_columns=list(dataset.features))
    
    return dataset

# def main():
#     from transformers import AutoTokenizer
#     model_id: str = "/home/ubuntu/LG3-interim-hf-weights"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     # if return_jsonl:
#     #     dataset = get_llamaguard_academic_dataset(None, tokenizer, "train", return_jsonl = True)
#     #     print(dataset[0:50])
#     # else:
#     #     dataset = get_llamaguard_academic_dataset(None, tokenizer, "train")
#     #     print(dataset[0])

#     dataset = get_llamaguard_academic_dataset(None, tokenizer, "train")
#     print(dataset[0])

# if __name__ == '__main__':
#     fire.Fire(main)
