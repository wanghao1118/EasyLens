from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


RADFM_QUICK_DEMO_ROOT = "/path/to/RadFM/Quick_demo"
RADFM_CHECKPOINT = "/path/to/RadFM/pytorch_model.bin"

LESION_PHRASES = [
    " lesion",
    " Lesion",
    " lesions",
    " Lesions",
    " nodule",
    " Nodule",
    " nodules",
    " Nodules",
    " nodular",
    " Nodular",
    " opacity",
    " Opacity",
    " opacities",
    " Opacities",
]

DENIAL_PHRASES = [
    " no",
    " No",
    " none",
    " None",
    " absent",
    " Absent",
    " without",
    " Without",
    " negative",
    " Negative",
]


@dataclass
class ReportRecord:
    sample_id: str
    image_rel: str
    image_path: Path
    prompt: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_prompt(prompt: str) -> str:
    prompt = prompt.replace("\r\n", "\n")
    if prompt.startswith("<image>\n"):
        return prompt.split("\n", 1)[1].strip()
    return prompt.replace("<image>", "").strip()


def strengthen_report_prompt(prompt: str, mode: str) -> str:
    prompt = clean_prompt(prompt)
    if mode == "benchmark_raw":
        return prompt
    if mode != "micro_focus_v3":
        raise ValueError(f"Unsupported prompt mode: {mode}")
    return (
        "Write a concise diagnostic chest CT report for this single image. "
        "Use exactly two sentences labeled Findings and Impression. "
        "Inspect carefully for any subtle tiny pulmonary nodule or focal nodular opacity. "
        "If any plausible tiny nodular finding is visible, explicitly describe it as a tiny pulmonary nodule or small pulmonary nodule and include an approximate size in mm. "
        "Use suspicious language if the finding is subtle. "
        "Do not add location unless clearly visible. "
        "Only say no focal pulmonary abnormality if you do not see any plausible focal nodular finding."
    )


def load_records(benchmark_dir: Path, prompt_mode: str, limit: int | None = None) -> list[ReportRecord]:
    data = json.loads((benchmark_dir / "vqa_report_llava.json").read_text(encoding="utf-8"))
    rows: list[ReportRecord] = []
    for item in data:
        rows.append(
            ReportRecord(
                sample_id=str(item["id"]),
                image_rel=str(item["image"]),
                image_path=benchmark_dir / str(item["image"]),
                prompt=strengthen_report_prompt(str(item["conversations"][0]["value"]), prompt_mode),
            )
        )
    return rows[:limit] if limit else rows


def get_tokenizer(processor_or_tokenizer: Any) -> Any:
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def _alpha_len(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def candidate_first_token_info(tokenizer: Any, phrases: list[str], min_alpha_len: int) -> tuple[list[int], dict[str, list[int]], dict[int, str]]:
    ids: list[int] = []
    phrase_map: dict[str, list[int]] = {}
    token_name_map: dict[int, str] = {}
    for phrase in phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        phrase_map[phrase] = [int(x) for x in token_ids]
        chosen: int | None = None
        token_name = ""
        for raw_id in token_ids:
            token_id = int(raw_id)
            try:
                token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            except Exception:
                token_name = str(token_id)
            decoded_piece = tokenizer.decode([token_id], skip_special_tokens=True)
            combined = token_name + decoded_piece
            if not any(ch.isalpha() for ch in combined):
                continue
            if _alpha_len(combined) >= min_alpha_len:
                chosen = token_id
            break
        if chosen is not None:
            ids.append(chosen)
            token_name_map[chosen] = token_name
    return list(dict.fromkeys(ids)), phrase_map, token_name_map


def decode_tokens(tokenizer: Any, token_ids: torch.Tensor) -> tuple[str, list[int], list[str]]:
    ids = [int(x) for x in token_ids.detach().cpu().tolist()]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        tokens = [str(x) for x in ids]
    return text, ids, tokens


def _at_least_once_prob(values: list[float]) -> float:
    if not values:
        return 0.0
    log_no_hit = 0.0
    for value in values:
        clipped = min(max(float(value), 0.0), 1.0 - 1e-12)
        log_no_hit += math.log1p(-clipped)
    return float(1.0 - math.exp(log_no_hit))


def analyze_scores(
    outputs: Any,
    generated_ids: torch.Tensor,
    tokenizer: Any,
    lesion_ids: list[int],
    denial_ids: list[int],
    topk_steps: int,
) -> dict:
    _text, gen_ids, gen_tokens = decode_tokens(tokenizer, generated_ids)
    scores = list(getattr(outputs, "scores", []) or [])
    if not scores or not gen_ids:
        return {
            "generated_token_count": len(gen_ids),
            "generated_token_ids": gen_ids,
            "generated_tokens": gen_tokens,
            "lesion_mass_mean": 0.0,
            "lesion_mass_max": 0.0,
            "lesion_mass_topk_mean": 0.0,
            "denial_mass_mean": 0.0,
            "denial_mass_max": 0.0,
            "denial_mass_topk_mean": 0.0,
            "margin_mean": 0.0,
            "margin_max": 0.0,
            "margin_topk_mean": 0.0,
            "lesion_sequence_prob": 0.0,
            "denial_sequence_prob": 0.0,
            "sequence_margin": 0.0,
            "lesion_beats_denial_step_fraction": 0.0,
            "positive_lesion_prob": 0.0,
        }

    lesion_idx = torch.tensor(lesion_ids, dtype=torch.long, device=scores[0].device)
    denial_idx = torch.tensor(denial_ids, dtype=torch.long, device=scores[0].device)
    lesion_masses: list[float] = []
    denial_masses: list[float] = []
    margins: list[float] = []
    for step_logits in scores:
        logits = step_logits[0].float()
        log_z = torch.logsumexp(logits, dim=-1)
        lesion_mass = torch.exp(torch.logsumexp(logits.index_select(0, lesion_idx), dim=-1) - log_z).item()
        denial_mass = torch.exp(torch.logsumexp(logits.index_select(0, denial_idx), dim=-1) - log_z).item()
        lesion_masses.append(float(lesion_mass))
        denial_masses.append(float(denial_mass))
        margins.append(float(lesion_mass - denial_mass))

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def topk_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        k = max(1, min(int(topk_steps), len(values)))
        return float(sum(sorted(values, reverse=True)[:k]) / k)

    lesion_sequence_prob = _at_least_once_prob(lesion_masses)
    denial_sequence_prob = _at_least_once_prob(denial_masses)
    lesion_wins = sum(1 for lesion, denial in zip(lesion_masses, denial_masses) if lesion > denial)
    return {
        "generated_token_count": len(gen_ids),
        "generated_token_ids": gen_ids,
        "generated_tokens": gen_tokens,
        "lesion_mass_mean": mean(lesion_masses),
        "lesion_mass_max": float(max(lesion_masses)),
        "lesion_mass_topk_mean": topk_mean(lesion_masses),
        "denial_mass_mean": mean(denial_masses),
        "denial_mass_max": float(max(denial_masses)),
        "denial_mass_topk_mean": topk_mean(denial_masses),
        "margin_mean": mean(margins),
        "margin_max": float(max(margins)),
        "margin_topk_mean": topk_mean(margins),
        "lesion_sequence_prob": lesion_sequence_prob,
        "denial_sequence_prob": denial_sequence_prob,
        "sequence_margin": float(lesion_sequence_prob - denial_sequence_prob),
        "positive_lesion_prob": float(lesion_sequence_prob * (1.0 - denial_sequence_prob)),
        "lesion_beats_denial_step_fraction": float(lesion_wins / len(lesion_masses)) if lesion_masses else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows]
        return float(sum(vals) / len(vals)) if vals else 0.0

    total = len(rows)
    return {
        "total": total,
        "lesion_sequence_prob_mean": mean("lesion_sequence_prob"),
        "denial_sequence_prob_mean": mean("denial_sequence_prob"),
        "sequence_margin_mean": mean("sequence_margin"),
        "positive_lesion_prob_mean": mean("positive_lesion_prob"),
        "lesion_sequence_gt_denial_sequence_rate": (
            float(sum(1 for row in rows if row["lesion_sequence_prob"] > row["denial_sequence_prob"]) / total)
            if total
            else 0.0
        ),
        "lesion_mass_mean_mean": mean("lesion_mass_mean"),
        "lesion_mass_max_mean": mean("lesion_mass_max"),
        "lesion_mass_topk_mean": mean("lesion_mass_topk_mean"),
        "denial_mass_mean_mean": mean("denial_mass_mean"),
        "denial_mass_max_mean": mean("denial_mass_max"),
        "denial_mass_topk_mean": mean("denial_mass_topk_mean"),
        "margin_mean_mean": mean("margin_mean"),
        "margin_max_mean": mean("margin_max"),
        "margin_topk_mean": mean("margin_topk_mean"),
    }


def force_greedy_generation_config(model: Any) -> None:
    """Remove sampling defaults so deterministic runs do not inherit model config noise."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    for name, value in {
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    }.items():
        if hasattr(generation_config, name):
            setattr(generation_config, name, value)


class MedGemmaRunner:
    def __init__(self, model_path: Path, device: str) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.tokenizer = get_tokenizer(self.processor)
        dtype = torch.bfloat16 if torch.cuda.is_available() and device != "cpu" else torch.float32
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() and device != "cpu" else None,
        )
        if device == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        force_greedy_generation_config(self.model)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_with_scores(self, image: Image.Image, image_path: Path, prompt_text: str, max_new_tokens: int) -> tuple[str, Any, torch.Tensor]:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        input_len = inputs["input_ids"].shape[1]
        generated = outputs.sequences[0][input_len:]
        text, _, _ = decode_tokens(self.tokenizer, generated)
        return text, outputs, generated


class LingshuRunner:
    def __init__(self, model_path: Path, device: str) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.tokenizer = get_tokenizer(self.processor)
        dtype = torch.bfloat16 if torch.cuda.is_available() and device != "cpu" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() and device != "cpu" else None,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        if device == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        force_greedy_generation_config(self.model)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_with_scores(self, image: Image.Image, image_path: Path, prompt_text: str, max_new_tokens: int) -> tuple[str, Any, torch.Tensor]:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if value.is_floating_point() and key in {"pixel_values", "image_grid_thw", "pixel_values_videos"}:
                    value = value.to(torch.bfloat16)
            moved[key] = value
        outputs = self.model.generate(
            **moved,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        input_len = moved["input_ids"].shape[1]
        generated = outputs.sequences[0][input_len:]
        text, _, _ = decode_tokens(self.tokenizer, generated)
        return text, outputs, generated


class HuluRunner:
    def __init__(self, model_path: Path, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.tokenizer = get_tokenizer(self.processor)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and device != "cpu" else torch.float32,
            device_map="auto" if torch.cuda.is_available() and device != "cpu" else None,
            attn_implementation="sdpa",
        )
        if device == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        force_greedy_generation_config(self.model)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_with_scores(self, image: Image.Image, image_path: Path, prompt_text: str, max_new_tokens: int) -> tuple[str, Any, torch.Tensor]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": str(image_path)}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        moved = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if key == "pixel_values":
                    value = value.to(torch.bfloat16)
            moved[key] = value
        outputs = self.model.generate(
            **moved,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        input_len = moved["input_ids"].shape[1]
        generated = outputs.sequences[0][input_len:] if outputs.sequences.shape[1] > input_len else outputs.sequences[0]
        text, _, _ = decode_tokens(self.tokenizer, generated)
        return text, outputs, generated


class LlavaRunner:
    def __init__(self, model_path: Path, device: str) -> None:
        import importlib.metadata as importlib_metadata
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        self.model_name = get_model_name_from_path(str(model_path))
        load_4bit = False
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(model_path),
            None,
            self.model_name,
            load_4bit=load_4bit,
            device_map="auto" if device != "cpu" else None,
        )
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def _candidate_conv_modes(self) -> list[str]:
        from llava.conversation import conv_templates

        name = self.model_name.lower()
        inferred = "mistral_instruct" if "mistral" in name else "llava_v1" if "v1" in name else "llava_v0"
        ordered: list[str] = []
        for mode in [inferred, "mistral_instruct", "llava_v1", "chatml_direct", "llava_v0"]:
            if mode in conv_templates and mode not in ordered:
                ordered.append(mode)
        return ordered

    @torch.no_grad()
    def generate_with_scores(self, image: Image.Image, image_path: Path, prompt_text: str, max_new_tokens: int) -> tuple[str, Any, torch.Tensor]:
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        qs = prompt_text
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        last_error: Exception | None = None
        for conv_mode in self._candidate_conv_modes():
            try:
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids = input_ids.unsqueeze(0).to(self.device)
                stop_str = conv.sep2 if conv.sep_style in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_2} else conv.sep
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sliced = outputs.sequences[0][input_ids.shape[1] :]
                text, _, _ = decode_tokens(self.tokenizer, sliced)
                generated = sliced
                if not text.strip():
                    full_sequence = outputs.sequences[0]
                    full_text, _, _ = decode_tokens(self.tokenizer, full_sequence)
                    if full_text.strip():
                        text = full_text
                        generated = full_sequence
                if stop_str and text.endswith(stop_str):
                    text = text[: -len(stop_str)].strip()
                return text, outputs, generated
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"LLaVA generation failed: {last_error}")


class RadFMRunner:
    def __init__(self, quick_demo_root: Path, checkpoint_path: Path, device: str) -> None:
        import importlib.util

        sys.path.insert(0, str(quick_demo_root))
        from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

        spec = importlib.util.spec_from_file_location("radfm_demo", quick_demo_root / "test.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        self.demo = mod
        self.tokenizer, self.image_padding_tokens = mod.get_tokenizer(tokenizer_path=str(quick_demo_root / "Language_files"))
        self.model = MultiLLaMAForCausalLM(lang_model_path=str(quick_demo_root / "Language_files")).half()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "pytorch_model.bin"
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        if device == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to(torch.float32).cpu()
        else:
            self.model = self.model.to(torch.float16).cuda()
        force_greedy_generation_config(self.model)
        force_greedy_generation_config(self.model.lang_model)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_with_scores(self, image: Image.Image, image_path: Path, prompt_text: str, max_new_tokens: int) -> tuple[str, Any, torch.Tensor]:
        del image
        image_list = [{"img_path": str(image_path), "position": 0}]
        text, vision_x = self.demo.combine_and_preprocess(prompt_text, image_list, self.image_padding_tokens)
        lang_x = self.tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"]
        lang_x = lang_x.to(torch.long).to(self.device)
        vision_x = vision_x.to(self.device)
        if torch.cuda.is_available() and self.device.type != "cpu":
            vision_x = vision_x.half()

        self.model.embedding_layer.flag = "Text"
        input_embedding, _ = self.model.embedding_layer(lang_x, vision_x)
        outputs = self.model.lang_model.generate(
            inputs_embeds=input_embedding,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        generated = outputs.sequences[0]
        text, _, _ = decode_tokens(self.tokenizer, generated)
        return text, outputs, generated


def build_runner(args: argparse.Namespace):
    if args.model_kind in {"medgemma", "medgemma1.5"}:
        return MedGemmaRunner(args.model_path, args.device)
    if args.model_kind == "lingshu":
        return LingshuRunner(args.model_path, args.device)
    if args.model_kind == "hulu-med-7b":
        return HuluRunner(args.model_path, args.device)
    if args.model_kind == "llava-med":
        return LlavaRunner(args.model_path, args.device)
    if args.model_kind == "radfm":
        return RadFMRunner(args.radfm_quick_demo_root, args.radfm_checkpoint, args.device)
    raise ValueError(f"Unsupported model_kind for token probability: {args.model_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report generation lesion-token probability evaluator for generative baselines.")
    parser.add_argument("--model-kind", choices=["llava-med", "hulu-med-7b", "lingshu", "medgemma", "medgemma1.5", "radfm"], required=True)
    parser.add_argument("--model-path", type=Path, required=False)
    parser.add_argument("--radfm-quick-demo-root", type=Path, default=Path(RADFM_QUICK_DEMO_ROOT))
    parser.add_argument("--radfm-checkpoint", type=Path, default=Path(RADFM_CHECKPOINT))
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=["benchmark_raw", "micro_focus_v3"], default="micro_focus_v3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--topk-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.model_kind != "radfm" and args.model_path is None:
        parser.error("--model-path is required unless --model-kind radfm")
    return args


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_dir(args.output_dir)
    records = load_records(args.benchmark_dir, args.prompt_mode, limit=(args.limit if args.limit > 0 else None))
    runner = build_runner(args)
    tokenizer = get_tokenizer(runner.tokenizer)
    lesion_ids, lesion_phrase_map, lesion_token_names = candidate_first_token_info(tokenizer, LESION_PHRASES, min_alpha_len=3)
    denial_ids, denial_phrase_map, denial_token_names = candidate_first_token_info(tokenizer, DENIAL_PHRASES, min_alpha_len=2)
    write_json(
        args.output_dir / "run_config.json",
        {
            "model_kind": args.model_kind,
            "model_path": str(args.model_path),
            "benchmark_dir": str(args.benchmark_dir),
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": int(args.max_new_tokens),
            "topk_steps": int(args.topk_steps),
            "total_records": len(records),
            "lesion_phrases": lesion_phrase_map,
            "lesion_first_token_ids": lesion_ids,
            "lesion_first_tokens": {str(k): v for k, v in lesion_token_names.items()},
            "denial_phrases": denial_phrase_map,
            "denial_first_token_ids": denial_ids,
            "denial_first_tokens": {str(k): v for k, v in denial_token_names.items()},
        },
    )
    pred_rows: list[dict] = []
    prob_rows: list[dict] = []
    for idx, rec in enumerate(records, start=1):
        image = Image.open(rec.image_path).convert("RGB")
        text, outputs, generated = runner.generate_with_scores(image, rec.image_path, rec.prompt, int(args.max_new_tokens))
        prob_meta = analyze_scores(outputs, generated, tokenizer, lesion_ids, denial_ids, int(args.topk_steps))
        pred_rows.append(
            {
                "sample_id": rec.sample_id,
                "image": rec.image_rel,
                "prompt": rec.prompt,
                "report_text": text,
            }
        )
        prob_rows.append(
            {
                "sample_id": rec.sample_id,
                "image": rec.image_rel,
                "prompt": rec.prompt,
                "report_text": text,
                **prob_meta,
            }
        )
        if idx % 10 == 0 or idx == len(records):
            print(f"[{args.model_kind}] report-token-prob {idx}/{len(records)}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {
        "benchmark_dir": str(args.benchmark_dir),
        "model_kind": args.model_kind,
        "prompt_mode": args.prompt_mode,
        "metrics": summarize(prob_rows),
    }
    write_jsonl(args.output_dir / "report_answers.jsonl", pred_rows)
    write_jsonl(args.output_dir / "lesion_token_prob_per_sample.jsonl", prob_rows)
    write_json(args.output_dir / "lesion_token_prob_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
