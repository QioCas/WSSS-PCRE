import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from utils.decoder_selftraining import (confident_labels,transform_pair,PseudoLabelDataset,
    initialize_heads,balanced_cross_entropy,anchor_kl,update_ema,refined_state)

class DecoderTests(unittest.TestCase):
    def test_subset_ids_confidence_and_independent_margin(self):
        probabilities=np.array([[[.1,.29,.4]],[[.85,.71,.6]],[[.05,0.,0.]]],dtype=np.float32)
        labels=confident_labels(probabilities,[0,3,7],confidence=.7,margin=.5)
        self.assertEqual(labels.tolist(),[[3,255,255]])
        one=confident_labels(np.ones((1,2,2)),[0])
        self.assertTrue(np.all(one==0))

    def test_all_ignore_loss_is_finite_and_differentiable(self):
        logits=torch.randn(2,3,4,4,requires_grad=True)
        loss=balanced_cross_entropy(logits,torch.full((2,4,4),255,dtype=torch.long))
        self.assertEqual(float(loss.detach()),0.);loss.backward()
        self.assertTrue(torch.equal(logits.grad,torch.zeros_like(logits)))

    def test_anchor_has_no_teacher_gradient_and_zero_self_distance(self):
        student=torch.randn(2,3,4,4,requires_grad=True)
        teacher=student.detach().clone().requires_grad_(True)
        loss=anchor_kl(student,teacher);self.assertLess(abs(float(loss.detach())),1e-6)
        loss.backward();self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_frozen_parameters_and_teacher_do_not_change(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(2,4);self.decoder=torch.nn.Linear(4,3)
        model=Toy();teacher,ema=initialize_heads(model)
        encoder_before=model.encoder.weight.detach().clone();teacher_before=teacher.weight.detach().clone()
        decoder_before=model.decoder.weight.detach().clone()
        optimizer=torch.optim.SGD(model.decoder.parameters(),lr=.1)
        features=model.encoder(torch.ones(4,2)).detach()
        logits=model.decoder(features)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,1]))+anchor_kl(logits[:,:,None,None],teacher(features)[:,:,None,None])
        optimizer.zero_grad();loss.backward();optimizer.step();update_ema(ema,model.decoder,.9)
        self.assertIsNone(model.encoder.weight.grad)
        self.assertTrue(torch.equal(model.encoder.weight,encoder_before))
        self.assertTrue(torch.equal(teacher.weight,teacher_before))
        self.assertFalse(torch.equal(model.decoder.weight,decoder_before))
        self.assertTrue(all(k.startswith('module.') for k in refined_state(model,ema)))

    def test_ema_updates_weights_without_updating_teacher(self):
        a=torch.nn.Linear(1,1,bias=False);b=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():a.weight.zero_();b.weight.fill_(2)
        update_ema(a,b,.75)
        self.assertAlmostEqual(a.weight.item(),.5)
        self.assertAlmostEqual(b.weight.item(),2.)

    def test_joint_geometry_keeps_rgb_and_labels_aligned(self):
        label=np.zeros((8,8),dtype=np.uint8);label[:,:4]=1
        image=np.repeat((label*255)[:,:,None],3,axis=2)
        result,target=transform_pair(Image.fromarray(image),Image.fromarray(label),random.Random(4),crop_size=12,short_range=(8,8),jitter=0)
        red=result[0].numpy()*58.395+123.675
        valid=target.numpy()!=255
        self.assertTrue(np.array_equal((red>127.5)[valid],target.numpy()[valid]==1))
        self.assertTrue((target==255).any())

    def test_training_dataset_works_without_any_pixel_gt_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'JPEGImages').mkdir();(root/'cache/raw_teacher').mkdir(parents=True)
            Image.fromarray(np.ones((8,8,3),dtype=np.uint8)*127).save(root/'JPEGImages/a.jpg')
            Image.fromarray(np.ones((8,8),dtype=np.uint8)*3).save(root/'cache/raw_teacher/a.png')
            plan={'data_folder':str(root),'crop_size':8,'resize_short_range':[8,8],'color_jitter':0.}
            data=PseudoLabelDataset(plan,root/'cache','raw_teacher',random.Random(0),names=['a'])
            image,label=data[0]
            self.assertEqual(tuple(image.shape),(3,8,8));self.assertTrue((label==3).all())
            self.assertFalse((root/'SegmentationClassAug').exists())

if __name__=='__main__':unittest.main()
