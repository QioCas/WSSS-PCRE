import random,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.precision_residual import (ResidualHead,correction_gate,strict_keep_kl,
    complementary_loss,veto_labels,negative_evidence,density_evidence,ResidualDataset)


class PrecisionResidualTests(unittest.TestCase):
    def test_zero_initialization_preserves_logits_and_gate_protects_background(self):
        torch.manual_seed(0);head=ResidualHead(4,8,limit=2.)
        base=torch.zeros(1,21,3,3);base[:,0,0]=20;base[:,2,1]=20
        gate=correction_gate(base)
        self.assertTrue((gate[:,:,0]==0).all());self.assertTrue((gate[:,:,1]==1).all())
        feature=torch.randn(1,4,3,3)
        self.assertTrue(torch.equal(base,base+gate*head(feature)))
        optimizer=torch.optim.SGD(head.parameters(),lr=.1)
        loss=torch.nn.functional.cross_entropy(base+gate*head(feature),torch.ones(1,3,3,dtype=torch.long))
        loss.backward();optimizer.step();correction=gate*head(feature)
        self.assertGreater(float(correction.abs().sum().detach()),0.)
        self.assertTrue((correction[:,:,0]==0).all());self.assertLessEqual(float(correction.abs().max().detach()),2.)

    def test_detail_head_zero_and_early_features_required(self):
        head=ResidualHead(4,8,detail=True)
        with self.assertRaises(ValueError):head(torch.randn(1,4,3,3))
        self.assertTrue((head(torch.randn(1,4,3,3),torch.randn(1,4,3,3))==0).all())

    def test_keep_kl_never_supervises_ignore_or_padding_or_conflict(self):
        base=torch.zeros(1,21,1,4,requires_grad=True)
        with torch.no_grad():base[:,1]=8
        logits=torch.randn_like(base,requires_grad=True)
        labels=torch.tensor([[[1,255,2,1]]]);valid=torch.tensor([[[True,True,True,False]]])
        loss,mask=strict_keep_kl(logits,base,labels,valid)
        self.assertEqual(mask.tolist(),[[[True,False,False,False]]]);loss.backward()
        self.assertIsNone(base.grad);self.assertTrue((logits.grad[:,:,:,1:]==0).all())

    def test_complementary_loss_lowers_only_rejected_class_without_hard_background_target(self):
        logits=torch.zeros(1,21,1,2,requires_grad=True)
        loss,mask=complementary_loss(logits,torch.tensor([[[2,255]]]),torch.ones(1,1,2,dtype=torch.bool))
        self.assertEqual(mask.tolist(),[[[True,False]]]);loss.backward()
        self.assertGreater(float(logits.grad[0,2,0,0]),0)
        self.assertTrue((logits.grad[:,:,:,1]==0).all())
        self.assertAlmostEqual(float(logits.grad[0,0,0,0]),float(logits.grad[0,1,0,0]))

    def test_veto_is_only_foreground_to_ignore_and_retains_class_for_negative_learning(self):
        labels=np.array([[0,1,2,255,3]],np.uint8);rejected=np.array([[0,1,3,2,255]],np.uint8)
        output,negative=veto_labels(labels,rejected)
        self.assertEqual(output.tolist(),[[0,255,2,255,3]])
        self.assertEqual(negative.tolist(),[[255,1,255,255,255]])

    def test_saturated_wrong_prediction_still_gets_finite_negative_gradient(self):
        logits=torch.zeros(1,21,1,1)
        logits[:,2]=1000.;logits.requires_grad_(True)
        loss,_=complementary_loss(logits,torch.tensor([[[2]]]),torch.ones(1,1,1,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertAlmostEqual(float(logits.grad[0,2,0,0]),1.,places=6)

    def test_bank_requires_positive_and_negative_class_support(self):
        features=torch.tensor([[[[1.]],[[0.]]]])
        positive=torch.zeros(21,1,2);negative=positive.clone();positive[:,0,1]=1;negative[:,0,0]=1
        bank={'positive':positive,'negative':negative,'positive_valid':torch.ones(21,dtype=torch.bool),
              'negative_valid':torch.ones(21,dtype=torch.bool)}
        bank['positive_valid'][2]=False
        mask=negative_evidence(features,bank,.1)
        self.assertFalse(bool(mask[0,0,0,0]));self.assertTrue(bool(mask[0,0,0,1]));self.assertFalse(bool(mask[0,0,0,2]))

    def test_density_likelihood_matches_diagonal_gaussian(self):
        features=torch.tensor([[[[1.]],[[0.]]]])
        bank={'projection':torch.eye(2),'gmm_means':torch.zeros(21,1,2),'gmm_variances':torch.ones(21,1,2),
              'gmm_weights':torch.ones(21,1),'density_cutoff':torch.full((21,),-2.),
              'density_valid':torch.ones(21,dtype=torch.bool)}
        # log N([1,0];0,I) = -0.5 - log(2*pi), below -2.
        mask=density_evidence(features,bank);self.assertTrue(bool(mask[0,0,0,1]));self.assertFalse(bool(mask[0,0,0,0]))
        bank['density_cutoff'].fill_(-3.);self.assertFalse(mask.equal(density_evidence(features,bank)))

    def test_dataset_joint_label_negative_geometry_without_pixel_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for d in ['JPEGImages','labels','negative']:(root/d).mkdir()
            image=np.ones((8,8,3),np.uint8)*127;label=np.ones((8,8),np.uint8)*2
            Image.fromarray(image).save(root/'JPEGImages/a.jpg')
            Image.fromarray(label).save(root/'labels/a.png');Image.fromarray(label).save(root/'negative/a.png')
            plan={'data_folder':str(root),'crop_size':12,'resize_short_range':[8,8],'jitter':0.}
            rgb,target,negative,valid=ResidualDataset(plan,root,['a'],random.Random(0))[0]
            self.assertTrue(torch.equal(target,negative));self.assertTrue(torch.equal(valid,target!=255))
            self.assertFalse((root/'SegmentationClassAug').exists());self.assertEqual(tuple(rgb.shape),(3,12,12))


if __name__=='__main__':unittest.main()
