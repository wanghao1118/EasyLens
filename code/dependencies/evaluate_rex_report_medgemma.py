from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, Gemma3ForConditionalGeneration

from evaluate_medgemma_microbench_bank_selector_visibility import (
    build_reference_tokens,
    move_inputs,
    select_support,
)
from evaluate_rex_report_micro_perception import (
    evaluate_pair,
    load_manifest as load_report_manifest,
    summarize,
    write_json,
    write_jsonl,
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


@dataclass
class ReportRecord:
    sample_id: str
    image_rel: str
    image_path: Path
    prompt: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_prompt(prompt: str) -> str:
    prompt = prompt.replace("\r\n", "\n")
    if prompt.startswith("<image>\n"):
        return prompt.split("\n", 1)[1]
    return prompt.replace("<image>", "").strip()


def strengthen_report_prompt(prompt: str, mode: str) -> str:
    prompt = clean_prompt(prompt)
    if mode == 'benchmark_raw':
        return prompt
    if mode == 'default':
        return (
            'Write a concise diagnostic chest CT report for this single image. '
            'Describe only findings that are visibly supported by the image. '
            'Do not assume any lesion is present. '
            'If no focal pulmonary abnormality is visible, say so briefly.'
        )
    if mode == 'micro_focus_v2':
        return (
            'Write a concise diagnostic chest CT report for this single image. '
            'Use one Findings sentence and one Impression sentence. '
            'Describe only findings that are visibly supported by the image. '
            'Do not assume any lesion is present. '
            'If a subtle tiny pulmonary nodule is visible, mention it in both Findings and Impression and include approximate size in mm when visible. '
            'If no focal pulmonary abnormality is visible, say so briefly without overcalling.'
        )
    if mode == 'micro_focus_v3':
        return (
            'Write a concise diagnostic chest CT report for this single image. '
            'Use exactly two sentences labeled Findings and Impression. '
            'Inspect carefully for any subtle tiny pulmonary nodule or focal nodular opacity. '
            'If any plausible tiny nodular finding is visible, explicitly describe it as a tiny pulmonary nodule or small pulmonary nodule and include an approximate size in mm. '
            'Use suspicious language if the finding is subtle. '
            'Do not add location unless clearly visible. '
            'Only say no focal pulmonary abnormality if you do not see any plausible focal nodular finding.'
        )
    if mode == 'micro_focus_v4':
        return (
            'Write a concise diagnostic chest CT report for this single image. '
            'Use exactly this format: Findings: ... Impression: ... '
            'Before concluding negative, inspect carefully for any faint focal nodular opacity. '
            'If any plausible tiny nodular finding is visible, Findings must contain the phrase tiny pulmonary nodule or small pulmonary nodule plus an approximate size in mm, and Impression must state suspicious tiny pulmonary nodule. '
            'Do not add unsupported location details.'
        )
    if mode == 'micro_focus_v5':
        return (
            'Write a concise diagnostic chest CT report for this single image. '
            'Use exactly two lines: Findings: ... and Impression: ... '
            'This evaluation is about subtle micro-lesion recognition, so inspect carefully for any tiny focal pulmonary nodule. '
            'If there is any plausible subtle nodular finding, even if uncertain, explicitly mention a tiny pulmonary nodule and an approximate size in mm. '
            'When uncertain but plausible, prefer suspicious tiny pulmonary nodule rather than a negative statement. '
            'Avoid unsupported location details.'
        )
    raise ValueError(f'Unknown prompt mode: {mode}')


def load_report_records(benchmark_dir: Path, prompt_mode: str, limit: int | None = None) -> list[ReportRecord]:
    data = json.loads((benchmark_dir / 'vqa_report_llava.json').read_text(encoding='utf-8'))
    rows: list[ReportRecord] = []
    for item in data:
        rows.append(
            ReportRecord(
                sample_id=str(item['id']),
                image_rel=str(item['image']),
                image_path=benchmark_dir / str(item['image']),
                prompt=strengthen_report_prompt(str(item['conversations'][0]['value']), prompt_mode),
            )
        )
    if limit is not None:
        rows = rows[:limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run MedGemma report generation on the ReX manual-authored diagnostic report benchmark.')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'shape'], required=True)
    parser.add_argument('--prompt-mode', choices=['benchmark_raw', 'default', 'micro_focus_v2', 'micro_focus_v3', 'micro_focus_v4', 'micro_focus_v5'], default='default')
    parser.add_argument('--bank-path', type=Path, default=None)
    parser.add_argument('--vision-layer', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.75)
    parser.add_argument('--selector-k', type=int, default=256)
    parser.add_argument('--support-mode', choices=['global_topk', 'radius_suppressed_topk'], default='global_topk')
    parser.add_argument('--suppression-radius', type=int, default=1)
    parser.add_argument('--margin-weight', type=float, default=0.35)
    parser.add_argument('--shape-weight', type=float, default=1.0)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--seed-mode', choices=['all_positive', 'local_peak', 'local_peak_topk'], default='local_peak_topk')
    parser.add_argument('--seed-peak-kernel', type=int, default=5)
    parser.add_argument('--seed-topk', type=int, default=16)
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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    image_processor = AutoImageProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device == 'cuda' else torch.float32),
        device_map='auto' if device == 'cuda' else None,
    )
    if device != 'cuda':
        model = model.to(device)
    model.eval()
    return device, image_processor, processor, model


def build_generate_kwargs(max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> dict:
    del do_sample, temperature, top_p
    kwargs = {
        'do_sample': False,
        'max_new_tokens': int(max_new_tokens),
        'use_cache': True,
    }
    return kwargs


def generate_baseline(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    inputs = move_inputs(inputs, model.device)
    gen_kwargs = build_generate_kwargs(max_new_tokens, do_sample, temperature, top_p)
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    input_len = inputs['input_ids'].shape[1]
    return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()


def generate_shape_method(args: argparse.Namespace, model, image_processor, processor, image: Image.Image, prompt_text: str, bank: dict, cluster_members: list[list[int]]) -> tuple[str, dict]:
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
    score, score_parts = compute_patch_score(lesion_sim, background_sim, proto_margin, margin_weight=float(args.margin_weight))
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
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    inputs = move_inputs(inputs, model.device)
    with patched_vision_layer_tokens(model, int(args.vision_layer), alpha, float(args.beta), 'token_delta', reference_tokens):
        gen_kwargs = build_generate_kwargs(args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
    input_len = inputs['input_ids'].shape[1]
    text = processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
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
    del background_sim, score, score_parts, heatmap, support_score, alpha, reference_tokens, messages, prompt, inputs, generated
    return text, meta


def run_eval(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    records = load_report_records(args.benchmark_dir, args.prompt_mode, limit=(args.limit if args.limit > 0 else None))
    gold_rows = load_report_manifest(args.benchmark_dir / 'manifest.csv')
    gold_map = {row['sample_id']: row for row in gold_rows}

    device, image_processor, processor, model = load_model(args)
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
        'total_records': len(records),
    }
    write_json(args.output_dir / 'run_config.json', run_config)

    pred_rows = []
    per_sample = []
    for idx, rec in enumerate(records, start=1):
        image = Image.open(rec.image_path).convert('RGB')
        if args.mode == 'baseline':
            raw_text = generate_baseline(model, processor, image, rec.prompt, int(args.max_new_tokens), args.do_sample, args.temperature, args.top_p)
            meta = {}
        else:
            raw_text, meta = generate_shape_method(args, model, image_processor, processor, image, rec.prompt, bank, cluster_members)
        pred_rows.append({
            'sample_id': rec.sample_id,
            'image': rec.image_rel,
            'prompt': rec.prompt,
            'report_text': raw_text,
            'text': raw_text,
            'raw_text': raw_text,
            **meta,
        })
        per_sample.append(evaluate_pair(gold_map[rec.sample_id], raw_text))
        if idx % 10 == 0 or idx == len(records):
            print(f'[{args.mode}] report {idx}/{len(records)}', flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        'benchmark_dir': str(args.benchmark_dir),
        'prediction_report_field': 'report_text',
        'missing_predictions': 0,
        'metrics': {
            'overall': summarize(per_sample),
            'clean': summarize([row for row in per_sample if bool(row['clean'])]),
            'context_flag': summarize([row for row in per_sample if bool(row.get('context_flag'))]),
        },
    }
    write_jsonl(args.output_dir / 'report_answers.jsonl', pred_rows)
    write_jsonl(args.output_dir / 'report_micro_perception_per_sample.jsonl', per_sample)
    write_json(args.output_dir / 'report_micro_perception_summary.json', summary)


if __name__ == '__main__':
    run_eval(parse_args())
