import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.representation_training import (select_trainable,student_features,
    reliable_labels,weak_strong_pair,ConsistencyDataset,consistency_loss)
from utils.sam_targets import choose_mask,compose_targets


class RepresentationTests(unittest.TestCase):
    def test_reliability_requires_confidence_and_view_agreement(self):
        probabilities=np.array([[[.1,.3,.1]],[[.9,.7,.9]]])
        votes=np.array([[[0,0,2]],[[6,6,4]]])
        labels,_,_=reliable_labels(probabilities,votes,[0,7],6,.8,5)
        self.assertEqual(labels.tolist(),[[7,255,255]])

    def test_selected_backbone_receives_gradient_through_student_features(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.patch_size=1
                self.blocks=torch.nn.ModuleList([torch.nn.Linear(3,3) for _ in range(4)])
                self.norm=torch.nn.LayerNorm(3)
            def forward_features(self,x):
                x=x.flatten(2).transpose(1,2)
                for block in self.blocks:x=block(x)
                return None,self.norm(x)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=Encoder();self.decoder=torch.nn.Conv2d(3,2,1)
            def to_2D(self,x,h,w):return x.transpose(1,2).reshape(-1,3,h,w)
        model=Model();selected=select_trainable(model,2)
        model.decoder(student_features(model,torch.randn(2,3,4,4))).square().mean().backward()
        self.assertIsNone(model.encoder.blocks[1].weight.grad)
        self.assertGreater(float(model.encoder.blocks[2].weight.grad.abs().sum()),0)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.blocks[:2].parameters()))
        self.assertTrue(any(k.startswith('encoder.blocks.2') for k in selected))
        select_trainable(model,0)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

    def test_cross_view_mask_excludes_cutout_padding_and_teacher_conflicts(self):
        student=torch.zeros(1,3,1,4,requires_grad=True)
        teacher=torch.tensor([[[[0.,0.,0.,0.]],[[8.,8.,8.,8.]],[[20.,20.,20.,20.]]]],requires_grad=True)
        loss,mask=consistency_loss(student,teacher,torch.tensor([[[1,255,0,1]]]),
            torch.tensor([[[True,True,True,False]]]),torch.tensor([[1.,0.]]))
        self.assertEqual(mask.tolist(),[[[True,True,False,False]]])
        loss.backward();self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(student.grad[:,:,:,2:],torch.zeros_like(student.grad[:,:,:,2:])))
        empty,_=consistency_loss(student,teacher,torch.zeros(1,1,4,dtype=torch.long),
            torch.zeros(1,1,4,dtype=torch.bool),torch.ones(1,2))
        self.assertEqual(float(empty.detach()),0.)

    def test_shared_geometry_and_training_without_ground_truth(self):
        plan={'crop_size':12,'resize_short_range':[8,8],'strong_jitter':0.,
              'grayscale_probability':0.,'blur_probability':0.,'cutout_probability':0.,'cutout_area':.1}
        label=np.zeros((8,8),np.uint8);label[:,:4]=1
        image=Image.fromarray(np.repeat((label*255)[:,:,None],3,2))
        weak,strong,target,support=weak_strong_pair(image,Image.fromarray(label),random.Random(2),plan)
        self.assertTrue(torch.equal(weak,strong));self.assertTrue(torch.equal(support,target!=255))
        red=weak[0].numpy()*58.395+123.675
        self.assertTrue(np.array_equal((red>127.5)[support],(target.numpy()==1)[support]))
        plan.update(cutout_probability=1.,crop_size=8)
        _,_,cut_target,cut_support=weak_strong_pair(image,Image.fromarray(label),random.Random(2),plan)
        self.assertLess(int(cut_support.sum()),64)
        self.assertTrue((cut_target[~cut_support]==255).all())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'JPEGImages').mkdir();(root/'labels').mkdir()
            image.save(root/'JPEGImages/a.jpg');Image.fromarray(label).save(root/'labels/a.png')
            np.save(root/'cls_labels_onehot.npy',{'a':np.ones(20)})
            plan.update(data_folder=str(root),list_folder=str(root))
            sample=ConsistencyDataset(plan,root,['a'],random.Random(0))[0]
            self.assertEqual(len(sample),5);self.assertFalse((root/'SegmentationClass').exists())

    def test_sam_filters_reject_quality_core_and_area_failures(self):
        p={'sam_min_score':.85,'sam_min_teacher_iou':.5,'sam_min_core_recall':.9,
           'sam_min_area_ratio':.5,'sam_max_area_ratio':2.}
        ref=np.zeros((20,20),bool);ref[5:15,5:15]=True;core=ref.copy()
        self.assertIsNone(choose_mask(np.stack([ref]),[.8],ref,core,p))
        self.assertIsNone(choose_mask(np.ones((1,20,20),bool),[.99],ref,core,p))
        small=ref.copy();small[5:10]=False
        self.assertIsNone(choose_mask(np.stack([small]),[.99],ref,core,p))
        self.assertIsNotNone(choose_mask(np.stack([ref]),[.9],ref,core,p))

    def test_sam_composition_protects_reliable_core_and_foreground_class(self):
        p={'sam_uncertain_radius':1,'sam_boundary_band':3}
        base=np.zeros((20,20),np.uint8);base[5:15,5:15]=1;base[8:12,15:18]=2
        ref=base==1;mask=ref.copy();mask[7:13,15:18]=True;mask[5,:]=False
        result,stats=compose_targets(base,[{'class_id':1,'mask':mask,'reference':ref}],p)
        self.assertTrue((result[8:12,15:18]==2).all())
        self.assertTrue((result[7:13,7:13]==1).all())
        self.assertEqual(stats['foreground_class_reassignments'],0)
        self.assertGreater(stats['new_ignore'],0)

    def test_conflicting_new_sam_foreground_becomes_ignore(self):
        p={'sam_uncertain_radius':1,'sam_boundary_band':4}
        base=np.zeros((20,20),np.uint8);left=np.zeros_like(base,bool);left[5:15,3:10]=True
        right=np.zeros_like(base,bool);right[5:15,10:17]=True
        wide=np.zeros_like(base,bool);wide[5:15,3:17]=True
        output,_=compose_targets(base,[{'class_id':1,'mask':wide,'reference':left},
            {'class_id':2,'mask':wide,'reference':right}],p)
        self.assertEqual(int(output[9,9]),255)


if __name__=='__main__':unittest.main()
