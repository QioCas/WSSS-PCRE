import unittest
import torch
import torch.nn as nn
from utils.sam_innovations import (
    compute_sam_patch_affinity,
    compute_boundary_weight_map,
    get_boundary_focal_seg_loss
)


class TestSAMInnovations(unittest.TestCase):
    def test_compute_sam_patch_affinity(self):
        B, H, W = 2, 448, 448
        h, w = 28, 28
        sam_mask = torch.zeros(B, H, W, dtype=torch.long)
        # Create a centered object not aligned with 16x16 grid (e.g. 100:300)
        sam_mask[0, 100:300, 100:300] = 15 # person
        pseudo_label_aux = torch.zeros(B, h, w, dtype=torch.long)
        pseudo_label_aux[0, 6:19, 6:19] = 15

        aff_mask, refined_aux = compute_sam_patch_affinity(
            sam_mask, pseudo_label_aux, purity_thresh=0.85, ignore_index=255
        )

        self.assertEqual(aff_mask.shape, (B, h * w, h * w))
        self.assertEqual(refined_aux.shape, (B, h, w))

        # Diagonal elements must be ignore_index (255)
        for i in range(h * w):
            self.assertEqual(aff_mask[0, i, i].item(), 255)

        # Boundary patches should be set to 255 in refined_aux
        boundary_count = (refined_aux[0] == 255).sum().item()
        self.assertGreater(boundary_count, 0, "Boundary patches should be flagged as 255")
        self.assertLess(boundary_count, h * w, "Not all patches should be boundary!")

    def test_compute_sam_patch_affinity_missing_cache_fallback(self):
        """Kiểm thử phòng vệ: Khi sam_mask rỗng/toàn 255, không được biến toàn bộ ảnh thành 255."""
        B, H, W = 2, 448, 448
        h, w = 28, 28
        sam_mask_missing = torch.full((B, H, W), 255, dtype=torch.long)
        pseudo_label_aux = torch.zeros(B, h, w, dtype=torch.long)
        pseudo_label_aux[0, 5:15, 5:15] = 7 # car
        pseudo_label_aux[1, 10:20, 10:20] = 12 # dog

        aff_mask, refined_aux = compute_sam_patch_affinity(
            sam_mask_missing, pseudo_label_aux, purity_thresh=0.85, ignore_index=255
        )

        # refined_aux phải bảo toàn pseudo_label_aux gốc
        self.assertTrue(torch.equal(refined_aux, pseudo_label_aux),
                        "Khi không có SAM mask, refined_aux phải fallback giữ nguyên pseudo_label_aux!")

        # aff_mask không được toàn bộ là 255 (phải có các cặp positive/negative)
        valid_pairs = (aff_mask[0] != 255).sum().item()
        self.assertGreater(valid_pairs, 0, "aff_mask không được là toàn 255 khi SAM thiếu!")

    def test_compute_boundary_weight_map(self):
        B, H, W = 2, 448, 448
        sam_mask = torch.zeros(B, H, W, dtype=torch.long)
        sam_mask[0, 100:300, 100:300] = 7 # car
        alpha = 2.5

        weight_map = compute_boundary_weight_map(sam_mask, alpha=alpha, kernel_size=9)
        self.assertEqual(weight_map.shape, (B, H, W))
        self.assertGreaterEqual(weight_map.min().item(), 1.0)
        self.assertAlmostEqual(weight_map.max().item(), 1.0 + alpha, delta=0.05)

    def test_get_boundary_focal_seg_loss(self):
        B, C, H, W = 2, 21, 448, 448
        pred = torch.randn(B, C, H, W, requires_grad=True)
        label = torch.randint(0, 21, (B, H, W))
        label[0, :30, :30] = 255

        weight_map = torch.ones(B, H, W)
        weight_map[0, 50:100, 50:100] = 3.0

        loss = get_boundary_focal_seg_loss(pred, label, weight_map, ignore_index=255)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())


if __name__ == '__main__':
    unittest.main()
