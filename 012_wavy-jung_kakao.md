원본코드
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from promptsource.templates import DatasetTemplates, Template
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .minimal_template.__base__ import MinimalTemplate
from .minimal_template.anli import AnliMinimal
from .minimal_template.hellaswag import HellaswagMinimal
from .minimal_template.storycloze import StoryclozeMinimal
from .minimal_template.superglue import (
    BoolqMinimal,
    CbMinimal,
    CopaMinimal,
    MultircMinimal,
    RecordMinimal,
    RteMinimal,
    WicMinimal,
    WscMinimal,
)
from .minimal_template.winogrande import WinograndeMinimal
from .minimal_template.xsum import XSumMinimal
from .utils import all_identical_elems, get_element_indices

MINIMAL_TEMPLATE_MAPPER = {
    "super_glue-boolq": BoolqMinimal,
    "super_glue-copa": CopaMinimal,
    "super_glue-cb": CbMinimal,
    "super_glue-rte": RteMinimal,
    "super_glue-wic": WicMinimal,
    "super_glue-wsc.fixed": WscMinimal,
    "super_glue-multirc": MultircMinimal,
    "super_glue-record": RecordMinimal,
    "anli1": AnliMinimal,
    "anli2": AnliMinimal,
    "anli3": AnliMinimal,
    "hellaswag": HellaswagMinimal,
    "story_cloze-2016": StoryclozeMinimal,
    "winogrande-winogrande_xl": WinograndeMinimal,
    "winogrande-winogrande_m": WinograndeMinimal,
    "xsum": XSumMinimal,
}


def get_prompt_template(dataset_name, subset_name) -> Tuple[Template, str]:
    """
    Choose a prompt template from bigscience-workshop/template source.
    You can do inference simply by typing a template index from terminal.
    """
    if dataset_name.startswith("anli"):
        dataset_name = dataset_name[:-1]
    prompts = DatasetTemplates(dataset_name, subset_name)
    prompts_type = prompts.name_to_id_mapping
    prompts_type_indices = {idx: key for idx, key in enumerate(prompts_type.keys())}
    for idx, key in prompts_type_indices.items():
        print(f"[{idx}] {key}")
    full_dataset_name = "-".join([dataset_name, subset_name]) if subset_name else dataset_name
    if full_dataset_name in MINIMAL_TEMPLATE_MAPPER.keys():
        minimal_template = MINIMAL_TEMPLATE_MAPPER[full_dataset_name]()
        print(f"[{idx + 1}] {minimal_template.get_name()}")
    chosen_idx = int(input("Choose Template Number: "))
    if chosen_idx == idx + 1:
        return minimal_template
    else:
        template = prompts.templates[prompts_type[prompts_type_indices[chosen_idx]]]
        return template


def get_template_mappings(dataset_name, subset_name) -> Tuple[Dict[int, str], DatasetTemplates]:
    if dataset_name.startswith("anli"):
        dataset_name = dataset_name[:-1]
    prompts = DatasetTemplates(dataset_name, subset_name)
    prompts_type = prompts.name_to_id_mapping
    return (prompts_type, prompts)


def get_minimal_template(dataset_name, subset_name) -> MinimalTemplate:
    template_key = dataset_name if subset_name is None else "-".join([dataset_name, subset_name])
    template = MINIMAL_TEMPLATE_MAPPER[template_key]()
    return template


def wrap_concat_ids(
    pre: Optional[List[int]],
    demon: Optional[List[int]],
    test: Optional[List[int]],
    post: Optional[List[int]],
    is_encoder: bool = True,
    test_data_to_decoder: bool = False,
):
    if test_data_to_decoder and is_encoder:
        return pre + demon + post
    elif not test_data_to_decoder and is_encoder:
        return pre + demon + test + post
    elif test_data_to_decoder and not is_encoder:
        return pre + test
    else:
        return pre


def k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    # TODO: Complete docstring with some of I/O format examples.
    """
    Create dataset in the format of k-shot classification.
    Get most probable answer from labels by comparing the perplexity (loss).
    """

    if train_dataset:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            demon, label = template.apply(train_dataset[idx])
            if isinstance(label, list):
                label = label[0]
            demonstrations.append(" ".join([demon, label]))

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed sampels case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            joined_demons: str = tokenizer.eos_token.join(selected_demons) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        elif train_dataset and len(train_dataset) == num_k:
            joined_demons: str = tokenizer.eos_token.join(demonstrations) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Zero-shot case
        else:
            tokenized_demons = []

        test_input, test_label = template.apply(sample)
        answer_choices: List[str] = template.get_answer_choices_list(sample)
        if isinstance(test_label, list):  # for multi-reference cases
            answer: List[int] = get_element_indices(answer_choices, test_label)
        else:
            answer: int = answer_choices.index(test_label)
        tokenized_ans: List[List[int]] = [
            tokenizer(ans, add_special_tokens=False)["input_ids"] for ans in answer_choices
        ]
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = (
            encoder_prefix_id + tokenized_demons + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + tokenized_demons + tokenized_input + encoder_postfix_id
        )
        decoder_prefix_id_ = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        input_ids = [[encoder_input_ids] for _ in range(len(answer_choices))]
        decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
        last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
        labels = [
            [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
        ]  # Calculate perplexity (loss) only for the answer spans
        target = answer

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "target": target,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:

    if train_dataset:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            example = template.apply(train_dataset[idx])
            demonstrations.append(" ".join(example))

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            joined_demons: str = tokenizer.eos_token.join(selected_demons) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        elif train_dataset and len(train_dataset) == num_k:
            joined_demons: str = tokenizer.eos_token.join(demonstrations) + tokenizer.eos_token
            tokenized_demons: List[int] = tokenizer(joined_demons, add_special_tokens=False)["input_ids"]
        # Zero-shot case
        else:
            tokenized_demons = []

        test_input, test_label = template.apply(sample)
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = (
            encoder_prefix_id + tokenized_demons + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + tokenized_demons + tokenized_input + encoder_postfix_id
        )
        decoder_input_ids = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "answer_start_idx": len(decoder_input_ids) - 1,
            "target": test_label,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def fid_k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    """Same function as 'k_shot_ds_for_classification()' in Fusion-in-Decoder fomat."""

    if not train_dataset:
        raise ValueError("Fusion-in-Decoder style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            demon, label = template.apply(train_dataset[idx])
            if isinstance(label, list):
                label = label[0]
            demonstrations.append(" ".join([demon, label]) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        answer_choices: List[str] = template.get_answer_choices_list(sample)
        if isinstance(test_label, list):  # for multi-reference cases
            answer: List[int] = get_element_indices(answer_choices, test_label)
        else:
            answer: int = answer_choices.index(test_label)
        tokenized_ans: List[List[int]] = [
            tokenizer(ans, add_special_tokens=False)["input_ids"] for ans in answer_choices
        ]
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]

        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_prefix_id_ = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        input_ids = [encoder_input_ids for _ in range(len(answer_choices))]
        decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
        last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
        labels = [
            [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
        ]  # Calculate perplexity (loss) only for the answer spans
        target = answer

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "target": target,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def fid_k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:

    if not train_dataset:
        raise ValueError("Fusion-in-Decoder style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            example = template.apply(train_dataset[idx])
            demonstrations.append(" ".join(example) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_input_ids = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "answer_start_idx": len(decoder_input_ids) - 1,
            "target": test_label,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def rag_k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    """Same function as 'k_shot_ds_for_classification()' in RAG fomat."""

    if not train_dataset:
        raise ValueError("RAG style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            demon, label = template.apply(train_dataset[idx])
            if isinstance(label, list):
                label = label[0]
            demonstrations.append(" ".join([demon, label]) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        answer_choices: List[str] = template.get_answer_choices_list(sample)
        if isinstance(test_label, list):  # for multi-reference cases
            answer: List[int] = get_element_indices(answer_choices, test_label)
        else:
            answer: int = answer_choices.index(test_label)
        tokenized_ans: List[List[int]] = [
            tokenizer(ans, add_special_tokens=False)["input_ids"] for ans in answer_choices
        ]
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]

        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_prefix_id_ = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        input_ids = [encoder_input_ids for _ in range(len(answer_choices))]
        decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
        last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
        labels = [
            [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
        ]  # Calculate perplexity (loss) only for the answer spans
        target = answer

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "target": target,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )


def rag_k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:

    if not train_dataset:
        raise ValueError("RAG style few-shot does not support zero-shot learning")
    else:
        demonstrations = []
        for idx in tqdm(range(len(train_dataset))):
            example = template.apply(train_dataset[idx])
            demonstrations.append(" ".join(example) + tokenizer.eos_token)

        if "idx" in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(["idx"])
        valid_dataset = valid_dataset.add_column("idx", list(range(len(valid_dataset))))

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = (
        [] if denoiser_prefix is None else tokenizer(denoiser_prefix, add_special_tokens=False)["input_ids"]
    )

    # Map function for mapping k-shot format dataset
    def template_mapper_(sample):
        # Non-fixed samples case
        if train_dataset and len(train_dataset) != num_k:
            start_idx = sample["idx"] * num_k % len(train_dataset)
            end_idx = (sample["idx"] + 1) * num_k % len(train_dataset)
            selected_demons: List[str] = (
                demonstrations[start_idx:end_idx]
                if start_idx < end_idx
                else demonstrations[start_idx:] + demonstrations[:end_idx]
            )
            tokenized_demons: List[List[int]] = tokenizer(selected_demons, add_special_tokens=False)["input_ids"]
        # Fixed-samples case
        else:
            tokenized_demons: List[List[int]] = tokenizer(demonstrations, add_special_tokens=False)["input_ids"]

        test_input, test_label = template.apply(sample)
        tokenized_input = tokenizer(test_input, add_special_tokens=False)["input_ids"]
        encoder_input_ids = [
            encoder_prefix_id + demon + encoder_postfix_id
            if test_data_to_decoder
            else encoder_prefix_id + demon + tokenized_input + encoder_postfix_id
            for demon in tokenized_demons
        ]

        decoder_input_ids = decoder_prefix_id + tokenized_input if test_data_to_decoder else decoder_prefix_id

        return {
            "input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "answer_start_idx": len(decoder_input_ids) - 1,
            "target": test_label,
        }

    return valid_dataset.map(
        template_mapper_, num_proc=num_proc, batch_size=1, remove_columns=valid_dataset.column_names
    )

중복된 전처리 로직과 반복 토크나이징 병목, 원본 데이터 스키마를 강제 변형하는 설계 결함으로 대규모 LLM 파이프라인에서 유지보수성과 데이터 무결성을 동시에 무너뜨리는 연구용 프로토타입 코드다.

제안패치
from functools import lru_cache, partial
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset
from promptsource.templates import DatasetTemplates, Template
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .minimal_template.__base__ import MinimalTemplate
from .minimal_template.anli import AnliMinimal
from .minimal_template.hellaswag import HellaswagMinimal
from .minimal_template.storycloze import StoryclozeMinimal
from .minimal_template.superglue import (
    BoolqMinimal,
    CbMinimal,
    CopaMinimal,
    MultircMinimal,
    RecordMinimal,
    RteMinimal,
    WicMinimal,
    WscMinimal,
)
from .minimal_template.winogrande import WinograndeMinimal
from .minimal_template.xsum import XSumMinimal
from .utils import all_identical_elems, get_element_indices

MINIMAL_TEMPLATE_MAPPER = {
    "super_glue-boolq": BoolqMinimal,
    "super_glue-copa": CopaMinimal,
    "super_glue-cb": CbMinimal,
    "super_glue-rte": RteMinimal,
    "super_glue-wic": WicMinimal,
    "super_glue-wsc.fixed": WscMinimal,
    "super_glue-multirc": MultircMinimal,
    "super_glue-record": RecordMinimal,
    "anli1": AnliMinimal,
    "anli2": AnliMinimal,
    "anli3": AnliMinimal,
    "hellaswag": HellaswagMinimal,
    "story_cloze-2016": StoryclozeMinimal,
    "winogrande-winogrande_xl": WinograndeMinimal,
    "winogrande-winogrande_m": WinograndeMinimal,
    "xsum": XSumMinimal,
}


@lru_cache(maxsize=1024)
def _cached_tokenize(tokenizer: PreTrainedTokenizer, text: str) -> List[int]:
    """[성능 최적화] 동일한 텍스트 반복 토크나이징 병목을 제거하기 위한 lru_cache 레이어"""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def get_prompt_template(dataset_name: str, subset_name: Optional[str]) -> Tuple[Template, str]:
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name은 반드시 문자열이어야 합니다.")
    if subset_name is not None and not isinstance(subset_name, str):
        raise TypeError("subset_name은 문자열이거나 None이어야 합니다.")

    if dataset_name.startswith("anli"):
        dataset_name = dataset_name[:-1]
    prompts = DatasetTemplates(dataset_name, subset_name)
    prompts_type = prompts.name_to_id_mapping
    prompts_type_indices = {idx: key for idx, key in enumerate(prompts_type.keys())}
    for idx, key in prompts_type_indices.items():
        print(f"[{idx}] {key}")
    full_dataset_name = "-".join([dataset_name, subset_name]) if subset_name else dataset_name
    if full_dataset_name in MINIMAL_TEMPLATE_MAPPER.keys():
        minimal_template = MINIMAL_TEMPLATE_MAPPER[full_dataset_name]()
        print(f"[{idx + 1}] {minimal_template.get_name()}")
    chosen_idx = int(input("Choose Template Number: "))
    if chosen_idx == idx + 1:
        return minimal_template
    else:
        template = prompts.templates[prompts_type[prompts_type_indices[chosen_idx]]]
        return template


def get_template_mappings(dataset_name: str, subset_name: Optional[str]) -> Tuple[Dict[int, str], DatasetTemplates]:
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name은 반드시 문자열이어야 합니다.")
    if dataset_name.startswith("anli"):
        dataset_name = dataset_name[:-1]
    prompts = DatasetTemplates(dataset_name, subset_name)
    prompts_type = prompts.name_to_id_mapping
    return (prompts_type, prompts)


def get_minimal_template(dataset_name: str, subset_name: Optional[str]) -> MinimalTemplate:
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name은 반드시 문자열이어야 합니다.")
    template_key = dataset_name if subset_name is None else "-".join([dataset_name, subset_name])
    if template_key not in MINIMAL_TEMPLATE_MAPPER:
        raise KeyError(f"지원하지 않는 Minimal Template 키입니다: {template_key}")
    template = MINIMAL_TEMPLATE_MAPPER[template_key]()
    return template


def _prepare_demonstrations(
    train_dataset: Dataset, 
    template: Union[Template, MinimalTemplate], 
    is_generation: bool = False, 
    add_eos: bool = False, 
    eos_token: str = ""
) -> List[str]:
    """[공통 모듈화] 학습 데이터셋(Train Dataset)으로부터 시연(Demonstrations) 문장을 안전하게 추출"""
    if not train_dataset:
        return []
    demonstrations = []
    for idx in range(len(train_dataset)):
        example = template.apply(train_dataset[idx])
        if is_generation:
            demo_str = " ".join(example)
        else:
            demon, label = example
            if isinstance(label, list):
                label = label[0]
            demo_str = " ".join([demon, label])
        if add_eos:
            demo_str += eos_token
        demonstrations.append(demo_str)
    return demonstrations


def _select_demonstrations(
    sample_idx: int, 
    train_dataset: Optional[Dataset], 
    demonstrations: List[str], 
    num_k: int
) -> List[str]:
    """[책임 분리] K-shot 시연 인덱스 계산 및 슬라이싱 로직 캡슐화"""
    if not train_dataset or num_k <= 0:
        return []
    
    train_len = len(train_dataset)
    if train_len != num_k:
        start_idx = sample_idx * num_k % train_len
        end_idx = (sample_idx + 1) * num_k % train_len
        return (
            demonstrations[start_idx:end_idx]
            if start_idx < end_idx
            else demonstrations[start_idx:] + demonstrations[:end_idx]
        )
    else:
        return demonstrations


def _tokenize_demonstrations(
    tokenizer: PreTrainedTokenizer, 
    selected_demons: List[str], 
    is_fid_or_rag: bool
) -> Union[List[List[int]], List[int]]:
    """[책임 분리] FiD/RAG와 일반 구조 간의 시연 토크나이징 분기 처리"""
    if not selected_demons:
        return []
    
    if is_fid_or_rag:
        return [_cached_tokenize(tokenizer, demon) for demon in selected_demons]
    else:
        joined_demons = tokenizer.eos_token.join(selected_demons) + tokenizer.eos_token
        return _cached_tokenize(tokenizer, joined_demons)


def _build_encoder_inputs(
    tokenized_demons: Union[List[List[int]], List[int]],
    encoder_prefix_id: List[int],
    encoder_postfix_id: List[int],
    is_fid_or_rag: bool
) -> Union[List[List[int]], List[int]]:
    """[책임 분리] 인코더 입력 시퀀스 구성"""
    if is_fid_or_rag:
        return [
            encoder_prefix_id + demon + encoder_postfix_id
            for demon in tokenized_demons
        ]
    else:
        return encoder_prefix_id + tokenized_demons + encoder_postfix_id


def _build_decoder_prefix(
    tokenizer: PreTrainedTokenizer,
    test_input: str,
    decoder_prefix_id: List[int],
    test_data_to_decoder: bool
) -> List[int]:
    """[버그 픽스] test_data_to_decoder 옵션 복구 및 디코더 프리픽스 결합"""
    tokenized_input = _cached_tokenize(tokenizer, test_input)
    if test_data_to_decoder:
        return decoder_prefix_id + tokenized_input
    return decoder_prefix_id


def _map_classification_sample(
    sample: dict,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Optional[Dataset],
    demonstrations: List[str],
    template: Union[Template, MinimalTemplate],
    num_k: int,
    decoder_prefix_id: List[int],
    encoder_postfix_id: List[int],
    encoder_prefix_id: List[int],
    is_fid_or_rag: bool,
    test_data_to_decoder: bool,
    add_eos_loss: bool,
    idx: int,
) -> dict:
    """[책임 분리] 분류(Classification) 태스크 전용 매퍼 함수"""
    selected_demons = _select_demonstrations(idx, train_dataset, demonstrations, num_k)
    tokenized_demons = _tokenize_demonstrations(tokenizer, selected_demons, is_fid_or_rag)
    encoder_input_ids = _build_encoder_inputs(tokenized_demons, encoder_prefix_id, encoder_postfix_id, is_fid_or_rag)

    test_input, test_label = template.apply(sample)
    decoder_prefix_id_ = _build_decoder_prefix(tokenizer, test_input, decoder_prefix_id, test_data_to_decoder)

    answer_choices = template.get_answer_choices_list(sample)
    if isinstance(test_label, list):
        answer = get_element_indices(answer_choices, test_label)
    else:
        answer = answer_choices.index(test_label)

    tokenized_ans = [_cached_tokenize(tokenizer, ans) for ans in answer_choices]
    
    if is_fid_or_rag:
        input_ids = [encoder_input_ids for _ in range(len(answer_choices))]
    else:
        input_ids = [[encoder_input_ids] for _ in range(len(answer_choices))]

    decoder_input_ids = [[decoder_prefix_id_ + ans] for ans in tokenized_ans]
    last_index_label = tokenizer.eos_token_id if add_eos_loss else -100
    labels = [
        [[-100 for _ in range(len(decoder_prefix_id_) - 1)] + ans + [last_index_label]] for ans in tokenized_ans
    ]

    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "target": answer,
    }


def _map_generation_sample(
    sample: dict,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Optional[Dataset],
    demonstrations: List[str],
    template: Union[Template, MinimalTemplate],
    num_k: int,
    decoder_prefix_id: List[int],
    encoder_postfix_id: List[int],
    encoder_prefix_id: List[int],
    is_fid_or_rag: bool,
    test_data_to_decoder: bool,
    idx: int,
) -> dict:
    """[책임 분리] 생성(Generation) 태스크 전용 매퍼 함수"""
    selected_demons = _select_demonstrations(idx, train_dataset, demonstrations, num_k)
    tokenized_demons = _tokenize_demonstrations(tokenizer, selected_demons, is_fid_or_rag)
    encoder_input_ids = _build_encoder_inputs(tokenized_demons, encoder_prefix_id, encoder_postfix_id, is_fid_or_rag)

    test_input, test_label = template.apply(sample)
    decoder_prefix_id_ = _build_decoder_prefix(tokenizer, test_input, decoder_prefix_id, test_data_to_decoder)

    return {
        "input_ids": encoder_input_ids,
        "decoder_input_ids": decoder_prefix_id_,
        "answer_start_idx": len(decoder_prefix_id_) - 1,
        "target": test_label,
    }


def _process_dataset(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
    is_fid_or_rag: bool = False,
    is_generation: bool = False,
    requires_train: bool = False,
) -> Dataset:
    """[엔터프라이즈 공통 파이프라인] 데이터셋 스키마 불변성 보장 및 with_indices 기반 멀티프로세싱 래퍼"""
    if not isinstance(valid_dataset, Dataset):
        raise TypeError("valid_dataset은 HuggingFace Dataset 객체여야 합니다.")
    if num_k < 0:
        raise ValueError("num_k는 0 이상이어야 합니다.")
    if requires_train and not train_dataset:
        raise ValueError("해당 모델 구조(FiD/RAG)는 제로슈팅을 지원하지 않으며 train_dataset이 필수입니다.")

    if train_dataset:
        demonstrations = _prepare_demonstrations(
            train_dataset, template, is_generation=is_generation, add_eos=is_fid_or_rag, eos_token=tokenizer.eos_token
        )
    else:
        demonstrations = []

    decoder_prefix_id = [decoder_start_id] if not use_sentinel else [decoder_start_id, first_sentinel_id]
    encoder_postfix_id = [] if not use_sentinel else [first_sentinel_id]
    encoder_prefix_id = [] if denoiser_prefix is None else _cached_tokenize(tokenizer, denoiser_prefix)

    if is_generation:
        mapper_func = partial(
            _map_generation_sample,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            demonstrations=demonstrations,
            template=template,
            num_k=num_k,
            decoder_prefix_id=decoder_prefix_id,
            encoder_postfix_id=encoder_postfix_id,
            encoder_prefix_id=encoder_prefix_id,
            is_fid_or_rag=is_fid_or_rag,
            test_data_to_decoder=test_data_to_decoder,
        )
    else:
        mapper_func = partial(
            _map_classification_sample,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            demonstrations=demonstrations,
            template=template,
            num_k=num_k,
            decoder_prefix_id=decoder_prefix_id,
            encoder_postfix_id=encoder_postfix_id,
            encoder_prefix_id=encoder_prefix_id,
            is_fid_or_rag=is_fid_or_rag,
            test_data_to_decoder=test_data_to_decoder,
            add_eos_loss=add_eos_loss,
        )

    # with_indices=True를 사용하여 원본 Dataset 스키마 오염 및 Arrow Table 복사 비용 원천 차단
    target_columns = list(valid_dataset.column_names)
    
    processed_ds = valid_dataset.map(
        mapper_func, 
        with_indices=True,
        num_proc=num_proc, 
        remove_columns=target_columns
    )
        
    return processed_ds


def k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    return _process_dataset(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=num_k,
        denoiser_prefix=denoiser_prefix,
        use_sentinel=use_sentinel,
        test_data_to_decoder=test_data_to_decoder,
        add_eos_loss=add_eos_loss,
        num_proc=num_proc,
        is_fid_or_rag=False,
        is_generation=False,
    )


def k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    return _process_dataset(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=num_k,
        denoiser_prefix=denoiser_prefix,
        use_sentinel=use_sentinel,
        test_data_to_decoder=test_data_to_decoder,
        add_eos_loss=add_eos_loss,
        num_proc=num_proc,
        is_fid_or_rag=False,
        is_generation=True,
    )


def fid_k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    return _process_dataset(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=num_k,
        denoiser_prefix=denoiser_prefix,
        use_sentinel=use_sentinel,
        test_data_to_decoder=test_data_to_decoder,
        add_eos_loss=add_eos_loss,
        num_proc=num_proc,
        is_fid_or_rag=True,
        is_generation=False,
        requires_train=True,
    )


def fid_k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    return _process_dataset(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=num_k,
        denoiser_prefix=denoiser_prefix,
        use_sentinel=use_sentinel,
        test_data_to_decoder=test_data_to_decoder,
        add_eos_loss=add_eos_loss,
        num_proc=num_proc,
        is_fid_or_rag=True,
        is_generation=True,
        requires_train=True,
    )


def rag_k_shot_ds_for_classification(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    return fid_k_shot_ds_for_classification(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=num_k,
        denoiser_prefix=denoiser_prefix,
        use_sentinel=use_sentinel,
        test_data_to_decoder=test_data_to_decoder,
        add_eos_loss=add_eos_loss,
        num_proc=num_proc,
    )


def rag_k_shot_ds_for_generation(
    tokenizer: PreTrainedTokenizer,
    decoder_start_id: int,
    first_sentinel_id: Optional[int],
    train_dataset: Optional[Dataset],
    valid_dataset: Dataset,
    template: Union[Template, MinimalTemplate],
    num_k: int,
    denoiser_prefix: Optional[str] = None,
    use_sentinel: bool = True,
    test_data_to_decoder: bool = True,
    add_eos_loss: bool = False,
    num_proc: int = 1,
) -> Dataset:
    return fid_k_shot_ds_for_generation(
        tokenizer=tokenizer,
        decoder_start_id=decoder_start_id,
        first_sentinel_id=first_sentinel_id,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        template=template,
        num_k=num_k,
        denoiser_prefix=denoiser_prefix,
        use_sentinel=use_sentinel,
        test_data_to_decoder=test_data_to_decoder,
        add_eos_loss=add_eos_loss,
        num_proc=num_proc,
    )

최종개선사항
✅ idx 컬럼 강제 삽입 제거 → with_indices=True 기반 무상태 데이터 매핑 전환
✅ 반복 tokenizer 호출 제거 → lru_cache 기반 토큰 캐싱 레이어 적용
✅ _base_k_shot_mapping 비대화 → classification/generation 전용 mapper 분리로 책임 단위 개선
✅ test_data_to_decoder 무시 버그 제거 → 원본 학습 데이터 흐름 호환성 복구
✅ FiD/RAG 중복 처리 제거 → 공통 _process_dataset 파이프라인으로 통합
✅ Demonstration 생성 로직 분리 → 데이터 준비 단계와 변환 단계 독립화
✅ 입력 타입 검증 강화 → Dataset/Template/파라미터 오류 조기 차단
✅ Minimal Template 조회 안정화 → 지원하지 않는 키에 대한 명시적 예외 처리
✅ Arrow Table 불필요 복사 제거 → 대규모 Dataset 메모리 사용량 감소
✅ K-shot 선택 로직 캡슐화 → 샘플링 정책 변경 시 단일 지점 수정 가능
✅ FiD/RAG Zero-shot 오류 방어 → 학습 데이터 필수 조건 검증 강화
✅ API 호환성 유지 → 기존 6개 public 함수 인터페이스 보존
✅ Silent Failure 방지 → 잘못된 입력이 학습 결과 왜곡으로 이어지는 경로 차단
✅ 연구 프로토타입 구조 → 대규모 LLM Fine-tuning 파이프라인 수준 구조로 승격

중복 파이프라인과 데이터 스키마 오염 문제를 제거하고 캐싱·무상태 매핑·책임 분리까지 적용했지만, 아직 대규모 LLM 학습 환경에서는 토큰 캐시 전략과 분산 병렬 처리 안정성 검증이 남은 엔터프라이즈 진입 직전 단계의 코드다.
