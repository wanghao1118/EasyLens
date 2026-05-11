from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from PIL import Image

from evaluate_rex_report_medgemma import build_generate_kwargs, ensure_dir, load_model, load_report_records
from evaluate_medgemma_microbench_bank_selector_visibility import (
    build_reference_tokens,
    move_inputs,
    select_support,
)
from evaluate_rex_shape_selector_hitmap import (
    build_cluster_member_lists,
    build_shape_heatmap,
    build_support_score,
    compute_patch_score,
    compute_position_background_maxsim,
    load_shape_bank,
    route_to_cluster_members,
)
from prototype_guided_fundus import extract_layer_tokens, patched_vision_layer_tokens
from evaluate_rex_report_micro_perception import write_json, write_jsonl


LESION_PHRASES = [
    ' lesion',
    ' Lesion',
    ' lesions',
    ' Lesions',
    ' nodule',
    ' Nodule',
    ' nodules',
    ' Nodules',
    ' nodular',
    ' Nodular',
    ' opacity',
    ' Opacity',
    ' opacities',
    ' Opacities',
]

DENIAL_PHRASES = [
    ' no',
    ' No',
    ' none',
    ' None',
    ' absent',
    ' Absent',
    ' without',
    ' Without',
    ' negative',
    ' Negative',
]


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
    parser.add_argument('--support-mode', choices=['global_topk', 'radius_suppressed_topk'], default='global_topk')
    parser.add_argument('--suppression-radius', type=int, default=1)
    parser.add_argument('--margin-weight', type=float, default=0.35)
    parser.add_argument('--shape-weight', type=float, default=1.0)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--seed-mode', choices=['all_positive', 'local_peak', 'local_peak_topk'], default='local_peak_topk')
    parser.add_argument('--seed-peak-kernel', type=int, default=5)
    parser.add_argument('--seed-topk', type=int, default=4)
    parser.add_argument('--base-score-mode', choices=['all_score', 'seed_only'], default='seed_only')
    parser.add_argument('--support-score-mode', choices=['shape_only', 'hybrid_max', 'hybrid_sum'], default='shape_only')
    parser.add_argument('--raw-score-weight', type=float, default=1.0)
    parser.add_argument('--gap-score-weight', type=float, default=0.0)
    parser.add_argument('--retrieval-topm', type=int, default=64)
    parser.add_argument('--retrieval-temperature', type=float, default=0.1)
    parser.add_argument('--prototype-query-chunk', type=int, default=512)
    parser.add_argument('--bg-bank-chunk', type=int, default=1024)
    parser.add_argument('--similarity-chunk', type=int, default=4096)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--do-sample', action='store_true', help='Deprecated; sampling is disabled for reproducible evaluation.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--topk-steps', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def get_tokenizer(processor):
    return getattr(processor, 'tokenizer', processor)


def _alpha_len(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def candidate_first_token_info(tokenizer, phrases: list[str], min_alpha_len: int) -> tuple[list[int], dict[str, list[int]], dict[int, str]]:
    ids = []
    phrase_map: dict[str, list[int]] = {}
    token_name_map: dict[int, str] = {}
    for phrase in phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        phrase_map[phrase] = [int(x) for x in token_ids]
        chosen = None
        chosen_name = ''
        for raw_id in token_ids:
            token_id = int(raw_id)
            token_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            decoded_piece = tokenizer.decode([token_id], skip_special_tokens=True)
            combined = token_name + decoded_piece
            if not any(ch.isalpha() for ch in combined):
                continue
            if _alpha_len(combined) >= min_alpha_len:
                chosen = token_id
                chosen_name = token_name
            break
        if chosen is not None:
            ids.append(chosen)
            token_name_map[chosen] = chosen_name
    deduped = list(dict.fromkeys(ids))
    return deduped, phrase_map, token_name_map


def prepare_generation_inputs(processor, image: Image.Image, prompt_text: str, device: torch.device):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    inputs = move_inputs(inputs, device)
    return prompt, inputs


def analyze_generation_scores(outputs, input_len: int, tokenizer, lesion_ids: list[int], denial_ids: list[int], topk_steps: int) -> dict:
    sequence = outputs.sequences[0][input_len:]
    generated_ids = [int(x) for x in sequence.tolist()]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    scores = list(outputs.scores)

    if not scores:
        return {
            'generated_token_count': 0,
            'generated_token_ids': generated_ids,
            'generated_tokens': generated_tokens,
            'lesion_mass_mean': 0.0,
            'lesion_mass_max': 0.0,
            'lesion_mass_topk_mean': 0.0,
            'denial_mass_mean': 0.0,
            'denial_mass_max': 0.0,
            'denial_mass_topk_mean': 0.0,
            'margin_mean': 0.0,
            'margin_max': 0.0,
            'margin_topk_mean': 0.0,
            'lesion_sequence_prob': 0.0,
            'denial_sequence_prob': 0.0,
            'sequence_margin': 0.0,
            'positive_lesion_prob': 0.0,
            'lesion_peak_step': None,
            'denial_peak_step': None,
            'lesion_peak_generated_token': None,
            'denial_peak_generated_token': None,
            'lesion_beats_denial_step_fraction': 0.0,
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

    def topk_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        k = max(1, min(int(topk_steps), len(values)))
        return float(sum(sorted(values, reverse=True)[:k]) / k)

    def at_least_once_prob(values: list[float]) -> float:
        if not values:
            return 0.0
        log_no_hit = 0.0
        for value in values:
            clipped = min(max(float(value), 0.0), 1.0 - 1e-12)
            log_no_hit += math.log1p(-clipped)
        return float(1.0 - math.exp(log_no_hit))

    lesion_peak_step = int(max(range(len(lesion_masses)), key=lambda i: lesion_masses[i]))
    denial_peak_step = int(max(range(len(denial_masses)), key=lambda i: denial_masses[i]))
    lesion_wins = sum(1 for l, d in zip(lesion_masses, denial_masses) if l > d)
    lesion_sequence_prob = at_least_once_prob(lesion_masses)
    denial_sequence_prob = at_least_once_prob(denial_masses)

    return {
        'generated_token_count': len(generated_ids),
        'generated_token_ids': generated_ids,
        'generated_tokens': generated_tokens,
        'lesion_mass_mean': float(sum(lesion_masses) / len(lesion_masses)),
        'lesion_mass_max': float(max(lesion_masses)),
        'lesion_mass_topk_mean': topk_mean(lesion_masses),
        'denial_mass_mean': float(sum(denial_masses) / len(denial_masses)),
        'denial_mass_max': float(max(denial_masses)),
        'denial_mass_topk_mean': topk_mean(denial_masses),
        'margin_mean': float(sum(margins) / len(margins)),
        'margin_max': float(max(margins)),
        'margin_topk_mean': topk_mean(margins),
        'lesion_sequence_prob': lesion_sequence_prob,
        'denial_sequence_prob': denial_sequence_prob,
        'sequence_margin': float(lesion_sequence_prob - denial_sequence_prob),
        'positive_lesion_prob': float(lesion_sequence_prob * (1.0 - denial_sequence_prob)),
        'lesion_peak_step': lesion_peak_step,
        'denial_peak_step': denial_peak_step,
        'lesion_peak_generated_token': generated_tokens[lesion_peak_step] if lesion_peak_step < len(generated_tokens) else None,
        'denial_peak_generated_token': generated_tokens[denial_peak_step] if denial_peak_step < len(generated_tokens) else None,
        'lesion_beats_denial_step_fraction': float(lesion_wins / len(lesion_masses)),
    }


@torch.no_grad()
def generate_baseline_with_probs(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, tokenizer, lesion_ids: list[int], denial_ids: list[int], topk_steps: int):
    _prompt, inputs = prepare_generation_inputs(processor, image, prompt_text, model.device)
    gen_kwargs = build_generate_kwargs(max_new_tokens, do_sample, temperature, top_p)
    gen_kwargs.update({'return_dict_in_generate': True, 'output_scores': True})
    outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs['input_ids'].shape[1]
    text = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
    prob_meta = analyze_generation_scores(outputs, input_len, tokenizer, lesion_ids, denial_ids, topk_steps)
    return text, prob_meta


def generate_shape_with_probs(args: argparse.Namespace, model, image_processor, processor, image: Image.Image, prompt_text: str, bank: dict, cluster_members: list[list[int]], tokenizer, lesion_ids: list[int], denial_ids: list[int], topk_steps: int):
    pixel_inputs = image_processor(images=image, return_tensors='pt')
    pixel_values = pixel_inputs['pixel_values'].to(device=model.device, dtype=(torch.bfloat16 if str(model.device).startswith('cuda') else torch.float32))
    source_tokens = extract_layer_tokens(model, pixel_values, int(args.vision_layer))
    token_norm = torch.nn.functional.normalize(source_tokens.float(), dim=-1)
    lesion_sim, best_member_idx, proto_margin = route_to_cluster_members(
        token_norm,
        bank,
        cluster_members,
        query_chunk=int(args.prototype_query_chunk),
    )
    background_sim = compute_position_background_maxsim(token_norm, bank, chunk_size=int(args.bg_bank_chunk))
    score, _score_parts = compute_patch_score(lesion_sim, background_sim, proto_margin, margin_weight=float(args.margin_weight))
    heatmap = build_shape_heatmap(
        score,
        lesion_sim,
        best_member_idx,
        bank,
        shape_weight=float(args.shape_weight),
        score_threshold=float(args.score_threshold),
        seed_mode=args.seed_mode,
        seed_peak_kernel=int(args.seed_peak_kernel),
        seed_topk=int(args.seed_topk),
        base_score_mode=args.base_score_mode,
    )
    support_score = build_support_score(
        score,
        heatmap,
        lesion_sim,
        background_sim,
        mode=args.support_score_mode,
        raw_score_weight=float(args.raw_score_weight),
        gap_score_weight=float(args.gap_score_weight),
    )
    alpha = select_support(
        support_score,
        support_mode=args.support_mode,
        k=int(args.selector_k),
        suppression_radius=int(args.suppression_radius),
    ).to(device=model.device)
    reference_tokens = build_reference_tokens(
        source_tokens=source_tokens,
        alpha=alpha,
        lesion_bank_raw=bank['lesion_bank'],
        lesion_bank_norm=bank['lesion_bank_norm'],
        lesion_delta_raw=None,
        topm=int(args.retrieval_topm),
        temperature=float(args.retrieval_temperature),
        similarity_chunk=int(args.similarity_chunk),
        retrieval_mode='reference_bank',
    )
    _prompt, inputs = prepare_generation_inputs(processor, image, prompt_text, model.device)
    gen_kwargs = build_generate_kwargs(args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
    gen_kwargs.update({'return_dict_in_generate': True, 'output_scores': True})
    with patched_vision_layer_tokens(model, int(args.vision_layer), alpha, float(args.beta), 'token_delta', reference_tokens):
        outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs['input_ids'].shape[1]
    text = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
    prob_meta = analyze_generation_scores(outputs, input_len, tokenizer, lesion_ids, denial_ids, topk_steps)
    meta = {
        'selected_patch_count': int((alpha > 0).sum().item()),
        'selected_patch_indices': [int(x) for x in torch.nonzero(alpha > 0, as_tuple=False).view(-1).tolist()],
        'score_max': float(score.max().item()),
        'score_mean': float(score.mean().item()),
        'heatmap_max': float(heatmap.max().item()),
        'heatmap_mean': float(heatmap.mean().item()),
        'support_score_max': float(support_score.max().item()),
        'support_score_mean': float(support_score.mean().item()),
    }
    del pixel_inputs, pixel_values, source_tokens, token_norm, lesion_sim, best_member_idx, proto_margin
    del background_sim, score, heatmap, support_score, alpha, reference_tokens, inputs, outputs
    return text, meta, prob_meta


def summarize_probabilities(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows]
        return float(sum(vals) / len(vals)) if vals else 0.0

    total = len(rows)
    return {
        'total': total,
        'lesion_mass_mean_mean': mean('lesion_mass_mean'),
        'lesion_mass_max_mean': mean('lesion_mass_max'),
        'lesion_mass_topk_mean': mean('lesion_mass_topk_mean'),
        'denial_mass_mean_mean': mean('denial_mass_mean'),
        'denial_mass_max_mean': mean('denial_mass_max'),
        'denial_mass_topk_mean': mean('denial_mass_topk_mean'),
        'margin_mean_mean': mean('margin_mean'),
        'margin_max_mean': mean('margin_max'),
        'margin_topk_mean': mean('margin_topk_mean'),
        'lesion_sequence_prob_mean': mean('lesion_sequence_prob'),
        'denial_sequence_prob_mean': mean('denial_sequence_prob'),
        'sequence_margin_mean': mean('sequence_margin'),
        'positive_lesion_prob_mean': mean('positive_lesion_prob'),
        'lesion_beats_denial_step_fraction_mean': mean('lesion_beats_denial_step_fraction'),
        'lesion_peak_gt_denial_peak_rate': float(sum(1 for row in rows if row['lesion_mass_max'] > row['denial_mass_max']) / total) if total else 0.0,
        'margin_max_positive_rate': float(sum(1 for row in rows if row['margin_max'] > 0.0) / total) if total else 0.0,
        'lesion_sequence_gt_denial_sequence_rate': float(sum(1 for row in rows if row['lesion_sequence_prob'] > row['denial_sequence_prob']) / total) if total else 0.0,
    }


def run_eval(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    records = load_report_records(args.benchmark_dir, args.prompt_mode, limit=(args.limit if args.limit > 0 else None))
    device, image_processor, processor, model = load_model(args)
    tokenizer = get_tokenizer(processor)
    lesion_ids, lesion_phrase_map, lesion_token_name_map = candidate_first_token_info(tokenizer, LESION_PHRASES, min_alpha_len=3)
    denial_ids, denial_phrase_map, denial_token_name_map = candidate_first_token_info(tokenizer, DENIAL_PHRASES, min_alpha_len=2)

    bank = None
    cluster_members = None
    if args.mode == 'shape':
        if args.bank_path is None:
            raise ValueError('--bank-path is required when --mode shape')
        bank = load_shape_bank(args.bank_path, torch.device(model.device))
        cluster_members = build_cluster_member_lists(bank)

    run_config = {
        'mode': args.mode,
        'model_path': str(args.model_path),
        'benchmark_dir': str(args.benchmark_dir),
        'bank_path': str(args.bank_path) if args.bank_path is not None else None,
        'vision_layer': int(args.vision_layer),
        'beta': float(args.beta),
        'selector_k': int(args.selector_k),
        'seed_topk': int(args.seed_topk),
        'support_mode': args.support_mode,
        'seed_mode': args.seed_mode,
        'support_score_mode': args.support_score_mode,
        'max_new_tokens': int(args.max_new_tokens),
        'device': device,
        'prompt_mode': args.prompt_mode,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'topk_steps': int(args.topk_steps),
        'total_records': len(records),
        'lesion_first_token_ids': lesion_ids,
        'lesion_first_tokens': {str(k): v for k, v in lesion_token_name_map.items()},
        'lesion_phrases': lesion_phrase_map,
        'denial_first_token_ids': denial_ids,
        'denial_first_tokens': {str(k): v for k, v in denial_token_name_map.items()},
        'denial_phrases': denial_phrase_map,
    }
    write_json(args.output_dir / 'run_config.json', run_config)

    pred_rows = []
    prob_rows = []
    for idx, rec in enumerate(records, start=1):
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
                lesion_ids,
                denial_ids,
                int(args.topk_steps),
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
                cluster_members,
                tokenizer,
                lesion_ids,
                denial_ids,
                int(args.topk_steps),
            )
        pred_rows.append({
            'sample_id': rec.sample_id,
            'image': rec.image_rel,
            'prompt': rec.prompt,
            'report_text': raw_text,
            **meta,
        })
        prob_rows.append({
            'sample_id': rec.sample_id,
            'image': rec.image_rel,
            'prompt': rec.prompt,
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
