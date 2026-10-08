from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # Allow discovery before optional ML dependencies are installed.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "challenge_neutral_kg"

if torch is not None:
    sys.path.insert(0, str(ROOT / "src"))
    from datasets import KNIIntentDataset, build_artifacts, kni_collate_fn
    from experiments.train_kcgt_contrastive import structured_contrastive_loss
    from models import build_kcgt_model


@unittest.skipUnless(
    torch is not None and (DATA_DIR / "train.jsonl").exists(),
    "Install the ML dependencies and generate data/processed/challenge_neutral_kg before running integration tests.",
)
class KCGTMethodUpgradeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.processed = DATA_DIR
        paths = [cls.processed / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
        cls.artifacts = build_artifacts(paths[0], paths, max_text_len=32)

    def test_relation_transformer_and_structured_loss_backward(self) -> None:
        dataset = KNIIntentDataset(self.processed / "train.jsonl", self.artifacts)
        batch = kni_collate_fn([dataset[index] for index in range(12)])
        model = build_kcgt_model(
            vocab_size=len(self.artifacts.token_to_id),
            num_classes=len(self.artifacts.class_to_id),
            pad_idx=self.artifacts.token_to_id["<pad>"],
            qos_dim=len(self.artifacts.qos_stats),
            node_feat_dim=len(self.artifacts.node_type_to_id),
            num_relations=len(self.artifacts.relation_to_id),
            num_kg_nodes=len(self.artifacts.node_to_id),
            graph_model="relation_transformer",
            graph_heads=4,
            relation_dim=16,
            text_dim=32,
            graph_dim=32,
            fusion_dim=48,
            projection_dim=24,
            dropout=0.0,
        )
        encoded = model.encode(batch)
        self.assertEqual(encoded["embedding"].shape, (12, 48))
        self.assertEqual(encoded["projection"].shape, (12, 24))
        loss, parts = structured_contrastive_loss(
            encoded["projection"],
            batch,
            temperature=0.15,
            positive_mode="group",
            kg_weight=0.40,
            qos_weight=0.35,
            slice_weight=0.25,
            hard_negative_boost=2.0,
            near_negative_boost=1.0,
            margin_weight=0.25,
            base_margin=0.10,
            margin_scale=0.20,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(parts["hard_pairs"]), 0.0)
        (loss + torch.nn.functional.cross_entropy(model.classifier(encoded["embedding"]), batch["y"])).backward()
        gradient = model.graph_encoder.relation_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_relation_ablation_is_deterministic(self) -> None:
        typed = KNIIntentDataset(self.processed / "train.jsonl", self.artifacts, relation_mode="typed")
        shuffled_a = KNIIntentDataset(self.processed / "train.jsonl", self.artifacts, relation_mode="shuffled", relation_seed=17)
        shuffled_b = KNIIntentDataset(self.processed / "train.jsonl", self.artifacts, relation_mode="shuffled", relation_seed=17)
        uniform = KNIIntentDataset(self.processed / "train.jsonl", self.artifacts, relation_mode="uniform")
        typed_rel = typed[0]["graph"].edge_type
        shuffled_rel_a = shuffled_a[0]["graph"].edge_type
        shuffled_rel_b = shuffled_b[0]["graph"].edge_type
        uniform_rel = uniform[0]["graph"].edge_type
        self.assertTrue(torch.equal(shuffled_rel_a, shuffled_rel_b))
        self.assertTrue(torch.equal(uniform_rel, torch.zeros_like(uniform_rel)))
        self.assertEqual(typed_rel.numel(), shuffled_rel_a.numel())


if __name__ == "__main__":
    unittest.main()
