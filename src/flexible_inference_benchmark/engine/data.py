# pylint: disable=too-many-positional-arguments
import abc
from typing import List, Tuple, Optional, Dict, Any
import logging
import json
import random
import os
from hashlib import sha256

from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore[attr-defined]
from flexible_inference_benchmark.engine import distributions

logger = logging.getLogger(__name__)


def get_data_end(data: List[int], tokenizer: PreTrainedTokenizerBase, idx: int, length: int, num_trials: int) -> int:
    assert length >= 0 and idx >= 0
    if length == 0:
        return idx

    idy = idx + length

    def get_length(x: int, y: int) -> int:
        return len(tokenizer.encode(tokenizer.decode(data[x:y])))

    for _ in range(num_trials):
        if get_length(idx, idy) == length:
            break
        if get_length(idx, idy) < length:
            idy += 1
        else:
            idy -= 1  # Could potentially be stuck in a cycle if the length is not achievable
            if idy < idx:
                idy = idx
                break

    if get_length(idx, idy) != length:
        logger.debug(f"Tried to achieve length {length} but failed. Achieved length {get_length(idx, idy)} instead")

    return idy


def hash_string(s: str) -> str:
    return sha256(s.encode()).hexdigest()


class Data(abc.ABC):
    IS_MULTIMODAL: bool = False

    @abc.abstractmethod
    def generate_data(self, size: int) -> List[Tuple[str, int, int, Optional[Dict[str, Any]]]]:
        pass


class Textfile(Data):
    def __init__(
        self,
        data: List[int],
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        num_trials: int,
        ignore_input_distribution: bool,
    ) -> None:
        self.prefix_str = prefix_str
        self.prefill_distribution = prefill_distribution
        self.output_token_distribution = output_token_distribution
        self.start_distribution = distributions.AdjustedUniformInt(0, len(data) - num_trials)
        self.tokenizer = tokenizer
        self.data = data
        self.num_trials = num_trials
        self.ignore_input_distribution = ignore_input_distribution

    @classmethod
    def with_prefix_str(
        cls,
        filename: str,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Textfile":
        with open(filename) as f:
            text = f.read()
        data = tokenizer.encode(text)

        return cls(
            data,
            prefix_str,
            prefill_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    @classmethod
    def with_prefix_len(
        cls,
        filename: str,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Textfile":
        with open(filename) as f:
            text = f.read()
        data = tokenizer.encode(text)

        if prefix_len + num_trials >= len(data):
            raise ValueError("Prefix length is too long")

        prefix_end = get_data_end(data, tokenizer, 0, prefix_len, num_trials)  # prefix real length

        prefix_str = tokenizer.decode(data[:prefix_end]) if prefix_end > 0 else ""

        return cls(
            data[prefix_end:],
            prefix_str,
            prefill_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    def generate_data(self, size: int) -> List[Tuple[str, int, int, Optional[Dict[str, Any]]]]:
        # Can save memory by using a generator. However for performance we will use a list
        input_data: List[Tuple[str, int, int, Optional[Dict[str, Any]]]] = []
        lengths = self.prefill_distribution.generate_distribution(size)
        output_tokens = self.output_token_distribution.generate_distribution(size)
        starts = self.start_distribution.generate_distribution(lengths)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        if self.ignore_input_distribution:
            # Add None for the multi-modal data part of the tuple
            input_data = [(self.prefix_str, prefix_len, output_tokens[i], None) for i in range(size)]
        else:
            for i in range(size):
                if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                    continue
                prompt_end = get_data_end(
                    self.data, self.tokenizer, starts[i], lengths[i] - prefix_len, self.num_trials
                )
                achieved_len = (prompt_end - starts[i]) + prefix_len

                input_data.append(
                    (
                        self.prefix_str + self.tokenizer.decode(self.data[starts[i] : prompt_end]),
                        achieved_len,
                        output_tokens[i],
                        None,  # Add None for multi-modal data
                    )
                )

        if len(input_data) < size:
            logger.debug(f"Generating {len(input_data)} requests instead of {size} requests.")
            return input_data

        return random.sample(input_data, size)


class Random(Data):
    def __init__(
        self,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        token_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        num_trials: int,
        ignore_input_distribution: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefill_distribution = prefill_distribution
        self.token_distribution = token_distribution
        self.output_token_distribution = output_token_distribution
        self.prefix_str = prefix_str
        self.num_trials = num_trials
        self.ignore_input_distribution = ignore_input_distribution

    @classmethod
    def with_prefix_str(
        cls,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Random":
        ## Specifying the middle 50% range to avoid accidental generation of <image> tokens
        token_distribution = distributions.UniformInt(
            len(tokenizer.get_vocab()) // 4, 3 * len(tokenizer.get_vocab()) // 4
        )

        return cls(
            prefix_str,
            prefill_distribution,
            token_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    @classmethod
    def with_prefix_len(
        cls,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Random":
        token_distribution = distributions.UniformInt(0, len(tokenizer.get_vocab()))
        data = list(token_distribution.generate_distribution(prefix_len + num_trials))
        prefix_end = get_data_end(data, tokenizer, 0, prefix_len, num_trials)  # prefix real length
        prefix_str = tokenizer.decode(data[:prefix_end]) if prefix_end > 0 else ""

        return cls(
            prefix_str,
            prefill_distribution,
            token_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    def generate_data(self, size: int) -> List[Tuple[str, int, int, Optional[Dict[str, Any]]]]:
        input_data: List[Tuple[str, int, int, Optional[Dict[str, Any]]]] = []
        lengths = self.prefill_distribution.generate_distribution(size)
        output_tokens = self.output_token_distribution.generate_distribution(size)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        if self.ignore_input_distribution:
            # Add None for the multi-modal data part of the tuple
            input_data = [(self.prefix_str, prefix_len, output_tokens[i], None) for i in range(size)]
        else:
            for i in range(size):
                data = list(self.token_distribution.generate_distribution(lengths[i] + self.num_trials))
                if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                    continue
                prompt_end = get_data_end(data, self.tokenizer, 0, lengths[i] - prefix_len, self.num_trials)
                achieved_len = prompt_end + prefix_len

                input_data.append(
                    (
                        self.prefix_str + self.tokenizer.decode(data[:prompt_end]),
                        achieved_len,
                        output_tokens[i],
                        None,  # Add None for multi-modal data
                    )
                )

        if len(input_data) < size:
            logger.debug(f"Generating {len(input_data)} requests instead of {size} requests.")
            return input_data
        return random.sample(input_data, size)


class ShareGPT(Data):
    def __init__(self, filename: str, tokenizer: PreTrainedTokenizerBase) -> None:
        # From https://github.com/vllm-project/vllm/blob/v0.4.0.post1/benchmarks/benchmark_serving.py#L310

        self.tokenizer = tokenizer

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [data for data in dataset if len(data["conversations"]) >= 2]

        tokenizer_id = tokenizer.name_or_path.replace("/", "_")
        cache_path = os.path.join(
            os.path.expanduser("~/.cache/flexible_inference_benchmark/"), f"sharegpt_sizes_{tokenizer_id}.json"
        )
        try:
            with open(cache_path, "r") as fcache:
                length_cache = json.load(fcache)
        except (FileNotFoundError, json.JSONDecodeError):
            length_cache = {}

        sequences_to_encode = [data["conversations"][0]["value"] for data in dataset] + [
            data["conversations"][1]["value"] for data in dataset
        ]
        all_in_cache = len(length_cache) > 0 and all(hash_string(seq) in length_cache for seq in sequences_to_encode)
        if not all_in_cache:
            encoded = tokenizer(sequences_to_encode)
            for i, seq in enumerate(sequences_to_encode):
                length_cache[hash_string(seq)] = len(encoded.input_ids[i])
            with open(cache_path, "w") as fcache:
                json.dump(length_cache, fcache)
        results_input_ids = [length_cache[hash_string(seq)] for seq in sequences_to_encode]
        tokenized_dataset = [
            (dataset[i]["conversations"][0]["value"], results_input_ids[i], results_input_ids[i + len(dataset)])
            for i in range(len(dataset))
        ]

        filtered_dataset = [
            (prompt_str, prompt_len, output_len)
            for prompt_str, prompt_len, output_len in tokenized_dataset
            if (prompt_len > 4 and output_len > 4)
        ]

        self.data = filtered_dataset

        logger.info("Loaded ShareGPT dataset.")

    def generate_data(self, size: int) -> List[Tuple[str, int, int, Optional[Dict[str, Any]]]]:
        sampled_data = []
        if len(self.data) < size:
            logger.debug(f"Generating {len(self.data)} requests instead of {size} requests.")
            sampled_data = self.data
        else:
            sampled_data = random.sample(self.data, size)
        
        # Add None for the multi-modal data part of the tuple
        return [(prompt, p_len, o_len, None) for prompt, p_len, o_len in sampled_data]


# -----------------------------------------------------------------------------
# ASR Dataset Implementation
# -----------------------------------------------------------------------------
class ASRDataset(Data):
    """
    Dataset class for processing a ASR dataset for transcription.
    Tested on the following set:

    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | Dataset        | Domain                                 | Speaking Style           | hf-subset                   |
    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | TED-LIUM       | TED talks                              | Oratory                  | release1, release2, release3|
    |                |                                        |                          | release3-speaker-adaptation |
    | VoxPopuli      | European Parliament                    | Oratory                  | en, de, it, fr,  ...        |
    | LibriSpeech    | Audiobook                              | Narrated                 | "LIUM/tedlium"              | # Note: Original patch had "openslr/librispeech_asr" here
    | GigaSpeech     | Audiobook, podcast, YouTube            | Narrated, spontaneous    | xs, s, m, l, xl, dev, test  |
    | SPGISpeech     | Financial meetings                     | Oratory, spontaneous     | S, M, L, dev, test          |
    | AMI            | Meetings                               | Spontaneous              | ihm, sdm                    |
    +----------------+----------------------------------------+--------------------------+-----------------------------+

    """  # noqa: E501
    IS_MULTIMODAL = True
    SUPPORTED_DATASET_PATHS = {
        "openslr/librispeech_asr", "facebook/voxpopuli", "LIUM/tedlium",
        "edinburghcstr/ami", "speechcolab/gigaspeech", "kensho/spgispeech"
    }

    DEFAULT_OUTPUT_LEN = 128  # As per patch 4/6
    # TODO Whisper-specific. Abstract interface when more models are supported.
    TRANSCRIPTION_PREAMBLE = "<|startoftranscript|><|en|><|transcribe|>"\
                              "<|notimestamps|>"
    skip_long_audios: bool = True

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_subset: Optional[str] = None,
        dataset_split: Optional[str] = "train", # Default to train as in patch
    ):
        from datasets import load_dataset  # type: ignore[attr-defined]
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split

        logger.info(
            f"Loading ASR dataset: {dataset_path}"
            f"{f' (subset: {dataset_subset})' if dataset_subset else ''}"
            f"{f' (split: {dataset_split})' if dataset_split else ''}"
        )
        try:
            self.data = load_dataset(dataset_path, name=dataset_subset, split=dataset_split, trust_remote_code=True)
            logger.info(f"Successfully loaded ASR dataset. Number of samples: {len(self.data)}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            # Fallback to an empty list or raise error, depending on desired behavior
            self.data = []
            raise  # Or handle more gracefully

    def generate_data(self, size: int) -> List[Tuple[str, int, int, Optional[Dict[str, Any]]]]:
        import librosa # type: ignore
        
        # Use the DEFAULT_OUTPUT_LEN from class attribute
        output_len = self.DEFAULT_OUTPUT_LEN
        
        prompt = ASRDataset.TRANSCRIPTION_PREAMBLE
        # Ensure tokenizer is available and prompt is not empty before tokenizing
        prompt_len = len(self.tokenizer.encode(prompt)) if self.tokenizer and prompt else 0

        sampled_requests: List[Tuple[str, int, int, Optional[Dict[str, Any]]]] = []
        skipped = 0

        if not self.data:
            logger.warning(f"ASR dataset {self.dataset_path} is empty or failed to load. Returning no requests.")
            return []

        # Ensure we don't try to sample more than available, or handle it by repeating/erroring
        num_to_sample = min(size, len(self.data))
        
        # Simple iteration for now, consider random sampling if `size` is much smaller than `len(self.data)`
        dataset_iterator = iter(self.data)

        for _ in range(num_to_sample):
            try:
                item = next(dataset_iterator)
            except StopIteration:
                logger.warning("Reached end of dataset sooner than expected.")
                break # No more items to sample

            # output_len = len(self.tokenizer._normalize(item['text'])) # Commented out as per patch 4/6
            
            audio_field = item.get("audio")
            if not audio_field or not isinstance(audio_field, dict):
                logger.warning(f"Skipping item due to missing or invalid 'audio' field: {item}")
                skipped +=1
                continue

            y, sr = audio_field.get("array"), audio_field.get("sampling_rate")
            if y is None or sr is None:
                logger.warning(f"Skipping item due to missing 'array' or 'sampling_rate' in audio field: {item}")
                skipped +=1
                continue
            
            # Ensure y is numpy array or compatible for librosa
            # y = np.asarray(y) # If needed, ensure numpy is imported

            duration_s = librosa.get_duration(y=y, sr=sr)
            # Whisper max supported duration
            if self.skip_long_audios and duration_s > 30:
                skipped += 1
                continue

            mm_content = {"audio": (y, sr)}
            sampled_requests.append(
                (
                    prompt,
                    prompt_len,
                    output_len, # Use fixed DEFAULT_OUTPUT_LEN
                    mm_content,
                )
            )
        
        if skipped > 0:
            logger.warning(
                f"{skipped} samples discarded from dataset {self.dataset_path} "
                "due to their length being greater than what Whisper supports or missing audio data."
            )
        
        # If fewer requests were generated than `size` due to filtering or small dataset,
        # the caller should handle this. Oversampling logic from patch is omitted.
        if len(sampled_requests) < size:
            logger.warning(
                f"Generated {len(sampled_requests)} requests for ASR, "
                f"but {size} were requested. Dataset might be too small or many samples were filtered."
            )
            # Potentially repeat samples if oversampling is desired, but patch omitted it too.

        return sampled_requests
