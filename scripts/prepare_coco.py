#!/usr/bin/env python3
"""Prepare COCO 2014 validation split lists, class labels, and ground-truth segmentation masks."""
import os
import sys
import zipfile
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_single_image(args):
    img_id, ann_file, out_dir = args
    # We load coco per worker or per batch
    pass

def main():
    ann_file = '/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE/dataset/COCO2014/annotations/instances_val2014.json'
    out_mask_dir = '/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE/dataset/COCO2014/coco/SegmentationClass/val'
    list_dir = '/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE/datasets/coco'
    preds_zip = '/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE/pretrained/coco_val_part_preds.zip'

    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(list_dir, exist_ok=True)

    print(f"Loading COCO annotations from {ann_file}...")
    coco = COCO(ann_file)
    cat_ids = sorted(list(coco.cats.keys()))
    cat_to_cls = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
    num_classes = len(cat_ids) # 80

    # 1. Get val_part image list from author's predictions
    print(f"Extracting val_part image list from {preds_zip}...")
    z = zipfile.ZipFile(preds_zip)
    val_part_names = sorted([x.replace('.png', '') for x in z.namelist() if x.endswith('.png')])
    print(f"Total val_part images: {len(val_part_names)}")

    # 2. Get full val image list
    all_img_ids = sorted(list(coco.imgs.keys()))
    all_val_names = [f"COCO_val2014_{img_id:012d}" for img_id in all_img_ids]
    print(f"Total full val images: {len(all_val_names)}")

    # Write list files
    with open(os.path.join(list_dir, 'val_part.txt'), 'w') as f:
        f.write('\n'.join(val_part_names) + '\n')
    with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(all_val_names) + '\n')
    print(f"Saved val_part.txt and val.txt into {list_dir}")

    # 3. Build cls_labels_onehot dict
    print("Generating cls_labels_onehot dictionary...")
    cls_labels = {}
    for img_id in all_img_ids:
        name = f"COCO_val2014_{img_id:012d}"
        onehot = np.zeros(num_classes, dtype=np.float32)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann['category_id'] in cat_to_cls:
                cls_idx = cat_to_cls[ann['category_id']] - 1
                onehot[cls_idx] = 1.0
        cls_labels[name] = onehot

    np.save(os.path.join(list_dir, 'cls_labels_onehot.npy'), cls_labels)
    print(f"Saved cls_labels_onehot.npy ({len(cls_labels)} items)")

    # 4. Generate GT segmentation masks for val_part images
    print(f"Generating GT segmentation masks for {len(val_part_names)} val_part images...")
    val_part_ids = [int(x.replace('COCO_val2014_', '')) for x in val_part_names]

    count = 0
    for img_id in tqdm(val_part_ids, desc="Generating masks"):
        img_info = coco.imgs[img_id]
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        # Sort small items first so they are drawn on top
        anns = sorted(anns, key=lambda x: x['area'])
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in cat_to_cls:
                cls_val = cat_to_cls[cat_id]
                m = coco.annToMask(ann)
                mask[m > 0] = cls_val

        # Save both short name (000000xxxxxx.png) and full name (COCO_val2014_000000xxxxxx.png)
        short_name = f"{img_id:012d}.png"
        full_name = f"COCO_val2014_{img_id:012d}.png"

        short_path = os.path.join(out_mask_dir, short_name)
        full_path = os.path.join(out_mask_dir, full_name)

        Image.fromarray(mask).save(short_path)
        if not os.path.exists(full_path):
            os.symlink(short_name, full_path)
        count += 1

    print(f"Successfully generated {count} GT masks in {out_mask_dir}!")

if __name__ == '__main__':
    main()
