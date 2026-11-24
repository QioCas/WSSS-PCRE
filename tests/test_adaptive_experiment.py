import tempfile
import unittest
from pathlib import Path

import torch

from utils.adaptive_threshold import AdaptiveCAMThreshold
from utils.pcre_experiment_state import atomic_save, rng_state, restore_rng, load_training_weights
from utils.optimizer import PolyWarmupAdamW


class AdaptiveTests(unittest.TestCase):
    def test_paired_checkpoint_continues_with_same_next_update(self):
        original = torch.nn.Linear(2, 1)
        restored = torch.nn.Linear(2, 1)
        settings = dict(lr=0.01, weight_decay=0, betas=(0.9, 0.999),
                        warmup_iter=1, max_iter=10, warmup_ratio=0.1, power=0.9)
        first = PolyWarmupAdamW(original.parameters(), **settings)
        original(torch.ones(1, 2)).sum().backward()
        first.step()
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'model.pth'
            training = Path(directory) / 'training.pth'
            atomic_save(original.state_dict(), weights)
            atomic_save({'model_filename': weights.name, 'optimizer': first.state_dict(),
                         'optimizer_global_step': first.global_step}, training)
            state = torch.load(training, weights_only=False)
            restored.load_state_dict(load_training_weights(state, training))
            second = PolyWarmupAdamW(restored.parameters(), **settings)
            second.load_state_dict(state['optimizer'])
            second.global_step = state['optimizer_global_step']
            for model, optimizer in ((original, first), (restored, second)):
                optimizer.zero_grad()
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
            self.assertTrue(torch.equal(original.weight, restored.weight))
            self.assertTrue(torch.equal(original.bias, restored.bias))
            self.assertEqual(first.param_groups[0]['lr'], second.param_groups[0]['lr'])

    def test_warmup_is_identity_and_scores_are_detached(self):
        tracker = AdaptiveCAMThreshold(2, warmup=10)
        cams = torch.tensor([[[[0.8, 0.7]], [[0.2, 0.3]]]], requires_grad=True)
        result = tracker.adapt(cams, torch.ones(1, 2), [(0, 1, 0, 2)], 0)
        self.assertTrue(torch.equal(result, cams))
        self.assertFalse(result.requires_grad)

    def test_padding_absent_classes_and_low_scores_do_not_update(self):
        tracker = AdaptiveCAMThreshold(3, momentum=0, warmup=0, ramp=0)
        cams = torch.tensor([[[[0.8, 0.99, 0.1]], [[0.3, 0.99, 0.1]], [[0.99, 0.99, 0.99]]]])
        tracker.update(cams, torch.tensor([[1, 1, 0]]), [(0, 1, 0, 1)])
        self.assertEqual(tracker.observations.tolist(), [1, 0, 0])
        self.assertAlmostEqual(tracker.global_confidence.item(), 0.8, places=6)
        self.assertAlmostEqual(tracker.thresholds(1)[2].item(), tracker.base, places=6)

    def test_empty_batch_preserves_state(self):
        tracker = AdaptiveCAMThreshold(2)
        before = {k: v.clone() for k, v in tracker.state_dict().items()}
        tracker.update(torch.zeros(1, 2, 2, 2), torch.ones(1, 2), [(0, 2, 0, 2)])
        for key, value in tracker.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_bounds_ramp_and_stronger_classes(self):
        tracker = AdaptiveCAMThreshold(2, momentum=0, warmup=2, ramp=2)
        tracker.update(torch.tensor([[[[0.9, 0.1]], [[0.1, 0.6]]]]),
                       torch.ones(1, 2), [(0, 1, 0, 2)])
        base = tracker.thresholds(2)
        middle, end = tracker.thresholds(3), tracker.thresholds(4)
        self.assertTrue(torch.allclose(middle, (base + end) / 2))
        self.assertGreater(end[0], end[1])
        self.assertTrue(((end >= 0.55) & (end <= 0.85)).all())

    def test_checkpoint_restores_optimizer_schedule_tracker_and_rng(self):
        model = torch.nn.Linear(2, 1)
        optimizer = PolyWarmupAdamW(model.parameters(), lr=0.01, weight_decay=0,
                                   betas=(0.9, 0.999), warmup_iter=1, max_iter=10,
                                   warmup_ratio=0.1, power=0.9)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        tracker = AdaptiveCAMThreshold(2)
        state = rng_state()
        expected_random = torch.rand(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'global_step': optimizer.global_step, 'adaptive': tracker.state_dict(),
                         'rng': state}, path)
            saved = torch.load(path, weights_only=False)
            copy = torch.nn.Linear(2, 1)
            copy.load_state_dict(saved['model'])
            other = PolyWarmupAdamW(copy.parameters(), lr=0.01, weight_decay=0,
                                   betas=(0.9, 0.999), warmup_iter=1, max_iter=10,
                                   warmup_ratio=0.1, power=0.9)
            other.load_state_dict(saved['optimizer'])
            other.global_step = saved['global_step']
            restore_rng(saved['rng'])
            self.assertTrue(torch.equal(expected_random, torch.rand(3)))
            self.assertTrue(torch.equal(model.weight, copy.weight))
            self.assertEqual(other.global_step, 1)
            self.assertTrue(other.state)
            tracker.load_state_dict(saved['adaptive'])
            with self.assertRaises(FileExistsError):
                atomic_save({}, path)


if __name__ == '__main__':
    unittest.main()
