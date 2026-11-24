"""Prepare PCRE split files and image-level class tags from VOC masks."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_ids(path):
    return [Path(line.split()[0]).stem for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_folder', type=Path, required=True)
    parser.add_argument('--train_list', type=Path, required=True,
                        help='The standard train_aug split (IDs or image/mask paths).')
    parser.add_argument('--output', type=Path, default=Path('datasets/voc'))
    args = parser.parse_args()
    train = read_ids(args.train_list)
    val = read_ids(args.data_folder / 'ImageSets/Segmentation/val.txt')
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        parser.error('Duplicate image IDs in a split')
    if set(train) & set(val):
        parser.error('Training and validation splits overlap')
    labels = {}
    for image_id in train + val:
        image_path = args.data_folder / 'JPEGImages' / (image_id + '.jpg')
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        mask_path = args.data_folder / 'SegmentationClassAug' / (image_id + '.png')
        with Image.open(mask_path) as image:
            classes = np.unique(np.asarray(image))
        classes = classes[(classes != 0) & (classes != 255)]
        if np.any(classes > 20):
            raise ValueError(f'Invalid VOC classes in {mask_path}: {classes}')
        onehot = np.zeros(20, dtype=np.float32)
        onehot[classes.astype(int) - 1] = 1
        labels[image_id] = onehot
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'train_aug.txt').write_text('\n'.join(train) + '\n')
    (args.output / 'val.txt').write_text('\n'.join(val) + '\n')
    np.save(args.output / 'cls_labels_onehot.npy', labels)
    print(f'Prepared {len(train)} train and {len(val)} validation images in {args.output}')


if __name__ == '__main__':
    main()
