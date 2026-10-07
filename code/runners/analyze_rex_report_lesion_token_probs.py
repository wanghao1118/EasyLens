from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from PIL import Image

from evaluate_rex_report_medgemma import build_generate_kwargs, ensure_dir, load_model, load_report_records
from evaluate_medgemma_microbench_bank_selector_visibility import move_inputs
from easylens_core import EasyLensConfig, compute_easylens, load_easylens_bank
from prototype_guided_fundus import extract_layer_tokens, patched_vision_layer_tokens
from evaluate_rex_report_micro_perception import write_json, write_jsonl


TARGET_STOPWORDS = {
    'a', 'an', 'and', 'approximately', 'are', 'both', 'for', 'in', 'is', 'measuring',
    'of', 'on', 'single', 'slice', 'the', 'there', 'this', 'to', 'up', 'visible', 'with',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze lesion-token probability mass during ReX report generation.')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'shape'], required=True)
    parser.add_argument('--prompt-mode', choices=['benchmark_raw', 'default', 'micro_focus_v2', 'micro_focus_v3', 'micro_focus_v4', 'micro_focus_v5'], default='micro_focus_v3')
    parser.add_argument('--bank-path', type=Path, default=None)
    parser.add_argument('--vision-layer', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--selector-k', type=int, default=64)
    parser.add_argument('--shape-weight', type=float, default=1.0)
    parser.add_argument('--seed-topk', type=int, default=4)
    parser.add_argument('--prototype-topm', type=int, default=1)
    parser.add_argument('--counterfactual-temperature', type=float, default=1.0)
    parser.add_argument('--amplifier-threshold', type=float, default=0.0)
    parser.add_argument('--similarity-chunk', type=int, default=4096)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--do-sample', action='store_true', help='Deprecated; sampling is disabled for reproducible evaluation.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--probe-steps', '--topk-steps', dest='probe_steps', type=int, default=3)
    parser.add_argument('--probe-vocab-topk', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def get_tokenizer(processor):
    return getattr(processor, 'tokenizer', processor)


def target_token_info(tokenizer, target_description: str) -> tuple[list[int], list[str], dict[int, str]]:
    """Create the strict per-sample lesion-token set required by Appendix B.2.

    Tokens come only from the benchmark target description. Common connective
    words, measurements, and boilerplate are excluded so the probe measures
    lesion category, morphology, anatomy, and clinically relevant descriptors.
    """

    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z-]+", target_description)]
    lesion_words = list(dict.fromkeys(word for word in words if word not in TARGET_STOPWORDS and len(word) > 2))
    token_ids: list[int] = []
    token_names: dict[int, str] = {}
    for word in lesion_words:
        for raw_id in tokenizer.encode(' ' + word, add_special_tokens=False):
            token_id = int(raw_id)
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if not any(character.isalpha() for character in decoded):
                continue
            token_ids.append(token_id)
            token_names[token_id] = tokenizer.convert_ids_to_tokens([token_id])[0]
    return list(dict.fromkeys(token_ids)), lesion_words, token_names


def prepare_generation_inputs(processor, image: Image.Image, prompt_text: str, device: torch.device):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    inputs = move_inputs(inputs, device)
    return prompt, inputs


def analyze_generation_scores(
    outputs,
    input_len: int,
    tokenizer,
    target_ids: list[int],
    probe_steps: int,
    vocab_topk: int,
    generated_sequence: torch.Tensor | None = None,
) -> dict:
    sequence = generated_sequence if generated_sequence is not None else outputs.sequences[0][input_len:]
    generated_ids = [int(x) for x in sequence.tolist()]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    scores = list(outputs.scores)[:max(1, int(probe_steps))]

    if not scores:
        return {
            'generated_token_count': 0,
            'generated_token_ids': generated_ids,
            'generated_tokens': generated_tokens,
            'probe_steps_used': 0,
            'lesion_mass_by_step': [],
            'strict_lesion_mass_by_step': [],
            'lesion_mass_mean': 0.0,
            'strict_lesion_mass_mean': 0.0,
            'probe_score': 0.0,
        }

    if not target_ids:
        raise ValueError('The paper-defined report probe requires a non-empty per-sample target lesion description.')
    target_idx = torch.tensor(target_ids, dtype=torch.long, device=scores[0].device)

    lesion_masses: list[float] = []
    strict_masses: list[float] = []
    for step_logits in scores:
        logits = step_logits[0].float()
        probabilities = torch.softmax(logits, dim=-1)
        lesion_masses.append(float(probabilities.index_select(0, target_idx).sum().item()))
        top_indices = torch.topk(probabilities, k=min(max(1, int(vocab_topk)), probabilities.numel())).indices
        in_topk = torch.isin(target_idx, top_indices)
        strict_masses.append(float(probabilities.index_select(0, target_idx[in_topk]).sum().item()))

    return {
        'generated_token_count': len(generated_ids),
        'generated_token_ids': generated_ids,
        'generated_tokens': generated_tokens,
        'probe_steps_used': len(scores),
        'lesion_mass_by_step': lesion_masses,
        'strict_lesion_mass_by_step': strict_masses,
        'lesion_mass_mean': float(sum(lesion_masses) / len(lesion_masses)),
        'strict_lesion_mass_mean': float(sum(strict_masses) / len(strict_masses)),
        'probe_score': float(100.0 * sum(strict_masses) / len(strict_masses)),
    }


@torch.no_grad()
def generate_baseline_with_probs(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, tokenizer, target_ids: list[int], probe_steps: int, probe_vocab_topk: int):
    _prompt, inputs = prepare_generation_inputs(processor, image, prompt_text, model.device)
    gen_kwargs = build_generate_kwargs(max_new_tokens, do_sample, temperature, top_p)
    gen_kwargs.update({'return_dict_in_generate': True, 'output_scores': True})
    outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs['input_ids'].shape[1]
    text = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
    prob_meta = analyze_generation_scores(outputs, input_len, tokenizer, target_ids, probe_steps, probe_vocab_topk)
    return text, prob_meta


def generate_shape_with_probs(args: argparse.Namespace, model, image_processor, processor, image: Image.Image, prompt_text: str, bank: dict, easylens_config: EasyLensConfig, tokenizer, target_ids: list[int], probe_steps: int, probe_vocab_topk: int):
    pixel_inputs = image_processor(images=image, return_tensors='pt')
    pixel_values = pixel_inputs['pixel_values'].to(device=model.device, dtype=(torch.bfloat16 if str(model.device).startswith('cuda') else torch.float32))
    source_tokens = extract_layer_tokens(model, pixel_values, int(args.vision_layer))
    reference_tokens, alpha, method_meta = compute_easylens(source_tokens, bank, easylens_config)
    _prompt, inputs = prepare_generation_inputs(processor, image, prompt_text, model.device)
    gen_kwargs = build_generate_kwargs(args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
    gen_kwargs.update({'return_dict_in_generate': True, 'output_scores': True})
    with patched_vision_layer_tokens(model, int(args.vision_layer), alpha, float(args.beta), 'paper_residual', reference_tokens):
        outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs['input_ids'].shape[1]
    text = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
    prob_meta = analyze_generation_scores(outputs, input_len, tokenizer, target_ids, probe_steps, probe_vocab_topk)
    del pixel_inputs, pixel_values, source_tokens, alpha, reference_tokens, inputs, outputs
    return text, method_meta, prob_meta


def summarize_probabilities(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows]
        return float(sum(vals) / len(vals)) if vals else 0.0

    total = len(rows)
    return {
        'total': total,
        'lesion_mass_mean_mean': mean('lesion_mass_mean'),
        'strict_lesion_mass_mean': mean('strict_lesion_mass_mean'),
        'probe_score': mean('probe_score'),
    }


def run_eval(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    records = load_report_records(args.benchmark_dir, args.prompt_mode, limit=(args.limit if args.limit > 0 else None))
    missing_targets = [record.sample_id for record in records if not record.target_description.strip()]
    if missing_targets:
        raise ValueError(
            f'Missing target_description for {len(missing_targets)} report samples. '
            'Prepare authorized local metadata with code/tools/prepare_rex_report_targets.py.'
        )
    device, image_processor, processor, model = load_model(args)
    tokenizer = get_tokenizer(processor)

    bank = None
    easylens_config = None
    if args.mode == 'shape':
        if args.bank_path is None:
            raise ValueError('--bank-path is required when --mode shape')
        bank = load_easylens_bank(args.bank_path, torch.device(model.device))
        easylens_config = EasyLensConfig(
            selector_k=int(args.selector_k),
            seed_topk=int(args.seed_topk),
            prototype_topm=int(args.prototype_topm),
            counterfactual_temperature=float(args.counterfactual_temperature),
            morphology_weight=float(args.shape_weight),
            amplifier_threshold=float(args.amplifier_threshold),
            similarity_chunk=int(args.similarity_chunk),
        )

    run_config = {
        'mode': args.mode,
        'model_path': str(args.model_path),
        'benchmark_dir': str(args.benchmark_dir),
        'bank_path': str(args.bank_path) if args.bank_path is not None else None,
        'vision_layer': int(args.vision_layer),
        'beta': float(args.beta),
        'selector_k': int(args.selector_k),
        'seed_topk': int(args.seed_topk),
        'shape_weight': float(args.shape_weight),
        'prototype_topm': int(args.prototype_topm),
        'counterfactual_temperature': float(args.counterfactual_temperature),
        'amplifier_threshold': float(args.amplifier_threshold),
        'method_profile': 'paper_equations_27_40' if args.mode == 'shape' else None,
        'seed_selection': 'global_topk' if args.mode == 'shape' else None,
        'candidate_selection': 'global_topk_calibrated_score' if args.mode == 'shape' else None,
        'reference_retrieval': 'cluster_nearest_neighbor' if args.mode == 'shape' else None,
        'residual_update': 'z_plus_alpha_w_reference_delta' if args.mode == 'shape' else None,
        'max_new_tokens': int(args.max_new_tokens),
        'device': device,
        'prompt_mode': args.prompt_mode,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'probe_steps': int(args.probe_steps),
        'probe_vocab_topk': int(args.probe_vocab_topk),
        'total_records': len(records),
        'bank_alignment': bank['alignment'] if bank is not None else None,
        'bank_schema_version': int(bank['schema_version']) if bank is not None else None,
    }
    write_json(args.output_dir / 'run_config.json', run_config)

    pred_rows = []
    prob_rows = []
    for idx, rec in enumerate(records, start=1):
        target_ids, target_words, target_token_names = target_token_info(tokenizer, rec.target_description)
        if not target_ids:
            raise ValueError(f'{rec.sample_id}: missing usable target_description for the Appendix B.2 probe')
        image = Image.open(rec.image_path).convert('RGB')
        if args.mode == 'baseline':
            raw_text, prob_meta = generate_baseline_with_probs(
                model,
                processor,
                image,
                rec.prompt,
                int(args.max_new_tokens),
                args.do_sample,
                args.temperature,
                args.top_p,
                tokenizer,
                target_ids,
                int(args.probe_steps),
                int(args.probe_vocab_topk),
            )
            meta = {}
        else:
            raw_text, meta, prob_meta = generate_shape_with_probs(
                args,
                model,
                image_processor,
                processor,
                image,
                rec.prompt,
                bank,
                easylens_config,
                tokenizer,
                target_ids,
                int(args.probe_steps),
                int(args.probe_vocab_topk),
            )
        pred_rows.append({
            'sample_id': rec.sample_id,
            'image': rec.image_rel,
            'prompt': rec.prompt,
            'target_description': rec.target_description,
            'report_text': raw_text,
            **meta,
        })
        prob_rows.append({
            'sample_id': rec.sample_id,
            'image': rec.image_rel,
            'prompt': rec.prompt,
            'target_description': rec.target_description,
            'target_words': target_words,
            'target_token_ids': target_ids,
            'target_tokens': {str(key): value for key, value in target_token_names.items()},
            'report_text': raw_text,
            **meta,
            **prob_meta,
        })
        if idx % 10 == 0 or idx == len(records):
            print(f'[{args.mode}] token-prob {idx}/{len(records)}', flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        'benchmark_dir': str(args.benchmark_dir),
        'mode': args.mode,
        'metrics': summarize_probabilities(prob_rows),
    }
    write_jsonl(args.output_dir / 'report_answers.jsonl', pred_rows)
    write_jsonl(args.output_dir / 'lesion_token_prob_per_sample.jsonl', prob_rows)
    write_json(args.output_dir / 'lesion_token_prob_summary.json', summary)


if __name__ == '__main__':
    run_eval(parse_args())
