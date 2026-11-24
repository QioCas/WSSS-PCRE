import unittest
from unittest import mock
import torch

from utils.adaptive_threshold import AdaptiveCAMThreshold
from utils.adaptive_consensus import CenteredCAMThreshold, expand_consensus_labels
from utils.camutils import refine_cams_with_bkg_v2
from utils.single_device import SingleDeviceModule


class ConsensusTests(unittest.TestCase):
    def test_single_device_checkpoint_and_alternating_unused_heads_match_direct_updates(self):
        direct=torch.nn.ModuleDict({'a':torch.nn.Linear(2,1),'b':torch.nn.Linear(2,1)})
        inner=torch.nn.ModuleDict({'a':torch.nn.Linear(2,1),'b':torch.nn.Linear(2,1)})
        inner.load_state_dict(direct.state_dict())
        wrapped=SingleDeviceModule(inner)
        optimizers=[torch.optim.SGD(direct.parameters(),lr=.1),torch.optim.SGD(wrapped.parameters(),lr=.1)]
        for branch in ['a','b','a']:
            for module,opt in zip([direct,wrapped.module],optimizers):
                opt.zero_grad(set_to_none=True)
                module[branch](torch.ones(2,2)).square().mean().backward();opt.step()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(direct.parameters(),wrapped.parameters())))
        self.assertTrue(all(key.startswith('module.') for key in wrapped.state_dict()))

    def test_single_rank_tracker_avoids_identity_collectives(self):
        tracker=CenteredCAMThreshold(2,momentum=0,warmup=0,ramp=0)
        with mock.patch('torch.distributed.is_initialized',return_value=True), \
             mock.patch('torch.distributed.get_world_size',return_value=1), \
             mock.patch('torch.distributed.all_reduce',side_effect=AssertionError('single rank must not communicate')):
            tracker.update(torch.tensor([[[[.8,.1]],[[.1,.6]]]]),torch.ones(1,2),[(0,1,0,2)])
        self.assertEqual(tracker.observations.tolist(),[1.,1.])

    def test_uniform_score_shift_does_not_relax_all_classes(self):
        cams=torch.tensor([[[[.6,.1]],[[.1,.6]]]])
        tags=torch.ones(1,2);boxes=[(0,1,0,2)]
        old=AdaptiveCAMThreshold(2,momentum=0,warmup=0,ramp=0)
        new=CenteredCAMThreshold(2,momentum=0,warmup=0,ramp=0,minimum=.65,maximum=.75)
        old.update(cams,tags,boxes);new.update(cams,tags,boxes)
        self.assertTrue(torch.all(old.thresholds(1)<.7))
        self.assertTrue(torch.allclose(new.thresholds(1),torch.full((2,),.7)))

    def test_admission_preserves_foreground_background_padding_and_rejects_conflicts(self):
        baseline=torch.tensor([[[2,0,255,255,255,255]]])
        low=torch.tensor([[[2,0,1,1,1,255]]])
        cams=torch.tensor([[[[.9,.9,.68,.68,.60,.8]],[[.1,.1,.1,.1,.1,.1]]]],requires_grad=True)
        proto=torch.tensor([[[[.9,.9,.8,.1,.8,.8]],[[.1,.1,.1,.9,.1,.1]]]])
        out,added=expand_consensus_labels(baseline,low,cams,proto,torch.ones(1,2),torch.tensor([.65,.75]))
        self.assertEqual(out.tolist(),[[[2,0,1,255,255,255]]])
        self.assertEqual(int(added.sum()),1)
        self.assertFalse(out.requires_grad)

    def test_warmup_and_unseen_classes_are_exact_baseline(self):
        tracker=CenteredCAMThreshold(2,warmup=8000,ramp=2000,minimum=.65,maximum=.75)
        tracker.update(torch.tensor([[[[.6,.1]],[[.1,.9]]]]),torch.ones(1,2),[(0,1,0,2)])
        for step in [0,7999,8000]:
            tau=tracker.thresholds(step)
            self.assertTrue(torch.allclose(tau,torch.full((2,),.7)))
            baseline=torch.full((1,1,2),255,dtype=torch.long)
            output,added=expand_consensus_labels(baseline,torch.ones_like(baseline),
                torch.ones(1,2,1,2),torch.ones(1,2,1,2),torch.ones(1,2),tau)
            self.assertTrue(torch.equal(output,baseline));self.assertFalse(added.any())
        empty=CenteredCAMThreshold(2,warmup=0,ramp=0)
        self.assertTrue(torch.equal(empty.thresholds(20),torch.full((2,),.7)))

    def test_absent_image_class_cannot_admit_pixels(self):
        labels=torch.full((1,1,1),255,dtype=torch.long)
        scores=torch.tensor([[[[.68]],[[1.]]]])
        proto=torch.tensor([[[[.1]],[[1.]]]])
        out,added=expand_consensus_labels(labels,torch.ones_like(labels)*2,scores,proto,
            torch.tensor([[1,0]]),torch.tensor([.65,.65]))
        self.assertTrue(torch.equal(out,labels));self.assertFalse(added.any())

    def test_refinement_details_keep_default_baseline_output(self):
        images=torch.rand(1,3,4,4)
        cams=torch.tensor([[[[.9,.9,.1,.1]]*4,[[.1,.1,.8,.8]]*4]])
        args=dict(ref_mod=lambda image,score:score,images=images,cams=cams,
                  cls_labels=torch.ones(1,2),high_thre=.7,low_thre=.25,
                  ignore_index=255,img_box=[(0,3,0,4)])
        original=refine_cams_with_bkg_v2(**args)
        baseline,high,low=refine_cams_with_bkg_v2(**args,return_details=True)
        self.assertTrue(torch.equal(original,baseline))
        self.assertTrue(torch.all(low[:,3:]==255))
        self.assertTrue(torch.all(baseline[:,3:]==255))

    def test_centered_bounds_and_checkpoint_state(self):
        tracker=CenteredCAMThreshold(3,momentum=0,warmup=0,ramp=0,minimum=.65,maximum=.75)
        tracker.update(torch.tensor([[[[.4,.1]],[[.1,.9]],[[1.,1.]]]]),
                       torch.tensor([[1,1,0]]),[(0,1,0,2)])
        tau=tracker.thresholds(1)
        self.assertTrue(((tau>=.65)&(tau<=.75)).all())
        self.assertAlmostEqual(tau[2].item(),.7,places=6)
        restored=CenteredCAMThreshold(3,momentum=0,warmup=0,ramp=0,minimum=.65,maximum=.75)
        restored.load_state_dict(tracker.state_dict())
        self.assertTrue(torch.equal(tau,restored.thresholds(1)))


if __name__=='__main__':unittest.main()
