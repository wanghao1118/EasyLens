from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code" / "runners"), str(ROOT / "code" / "method"), str(ROOT / "code" / "dependencies")]

from analyze_rex_report_lesion_token_probs import analyze_generation_scores
from easylens_core import EasyLensConfig, compute_easylens, load_easylens_bank
from prototype_guided_fundus import build_modified_final_tokens_from_alpha


class DummyTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"token_{int(token_id)}" for token_id in token_ids]


class EasyLensCoreTest(unittest.TestCase):
    def _bank_payload(self) -> dict:
        return {
            "schema_version": 2,
            "grid_side": 3,
            "feature_dim": 2,
            "lesion_bank": np.asarray([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]], dtype=np.float16),
            "lesion_prototypes": np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float16),
            "lesion_cluster_ids": np.asarray([0, 0, 1], dtype=np.int16),
            "lesion_cluster_member_order": np.asarray([0, 1, 2], dtype=np.int32),
            "lesion_cluster_offsets": np.asarray([0, 2, 3], dtype=np.int32),
            "morphology_priors": np.asarray(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                dtype=np.float16,
            ),
            "background_position_bank": np.tile(np.asarray([[0.0, 1.0]], dtype=np.float16), (9, 1)),
            "background_position_counts": np.ones(9, dtype=np.int32),
            "background_position_offsets": np.arange(10, dtype=np.int64),
            "background_global_bank": np.asarray([[0.0, 1.0]], dtype=np.float16),
        }

    def test_schema_v2_inference_uses_continuous_morphology_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            bank_path = Path(temporary_directory) / "bank.pt"
            torch.save(self._bank_payload(), bank_path)
            bank = load_easylens_bank(bank_path, torch.device("cpu"))

        source_tokens = torch.tensor(
            [[0.95, 0.05], [0.90, 0.10], [0.10, 0.90]] + [[0.0, 1.0]] * 6,
            dtype=torch.float32,
        )
        config = EasyLensConfig(
            selector_k=1,
            seed_topk=2,
            prototype_topm=1,
            counterfactual_temperature=1.0,
            morphology_weight=1.0,
            amplifier_threshold=0.0,
        )
        references, weights, metadata = compute_easylens(source_tokens, bank, config)

        expected_first_score = torch.sigmoid(
            torch.cosine_similarity(source_tokens[0], torch.tensor([1.0, 0.0]), dim=0)
            - torch.cosine_similarity(source_tokens[0], torch.tensor([0.0, 1.0]), dim=0)
        ).item()
        self.assertEqual(metadata["method_profile"], "paper_equations_27_40")
        self.assertEqual(metadata["seed_selection"], "global_topk")
        self.assertEqual(metadata["seed_patch_indices"], [0, 1])
        self.assertEqual(metadata["reference_retrieval"], "cluster_nearest_neighbor")
        self.assertEqual(metadata["residual_update"], "z_plus_alpha_w_reference_delta")
        self.assertEqual(metadata["bank_alignment"], "equation_literal_v3")
        self.assertEqual(metadata["selected_patch_count"], 1)
        self.assertGreater(metadata["amplified_patch_count"], 1)
        self.assertAlmostEqual(metadata["initial_score_max"], expected_first_score, places=6)
        self.assertTrue(torch.any((weights > 0.0) & (weights != 1.0)))
        active = torch.nonzero(weights > 0.0, as_tuple=False).view(-1)
        self.assertTrue(torch.all(references[active, 0] > 0.0))
        self.assertTrue(torch.equal(references[0], torch.tensor([1.0, 0.0])))

    def test_paper_residual_applies_unmodified_morphology_weight(self):
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        reference = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)
        weight = torch.tensor([0.25, 0.0], dtype=torch.float32)

        result = build_modified_final_tokens_from_alpha(
            source,
            weight,
            beta=2.0,
            mode="paper_residual",
            reference_tokens=reference,
        )

        expected = torch.tensor([[0.5, 1.0], [0.0, 1.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(result["modified_final_tokens"], expected))

    def test_report_probe_uses_first_steps_and_vocabulary_topk_intersection(self):
        logits = [
            torch.tensor([[0.0, 1.0, 5.0, -1.0]], dtype=torch.float32),
            torch.tensor([[6.0, 1.0, 0.0, -1.0]], dtype=torch.float32),
        ]
        outputs = SimpleNamespace(
            sequences=torch.tensor([[2, 0]], dtype=torch.long),
            scores=tuple(logits),
        )
        result = analyze_generation_scores(
            outputs,
            0,
            DummyTokenizer(),
            target_ids=[2],
            probe_steps=2,
            vocab_topk=1,
        )

        first_probability = float(torch.softmax(logits[0][0], dim=-1)[2].item())
        self.assertEqual(result["probe_steps_used"], 2)
        self.assertAlmostEqual(result["strict_lesion_mass_by_step"][0], first_probability, places=6)
        self.assertEqual(result["strict_lesion_mass_by_step"][1], 0.0)
        self.assertAlmostEqual(result["probe_score"], 50.0 * first_probability, places=5)


if __name__ == "__main__":
    unittest.main()
