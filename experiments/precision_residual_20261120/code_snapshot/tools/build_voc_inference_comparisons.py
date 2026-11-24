"""Build inspectable four-column VOC comparisons from completed pair inference."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.infer_voc_checkpoint_pair import palette, hist_for, image_score

FONT_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT=ImageFont.truetype(FONT_PATH,19)
SMALL=ImageFont.truetype(FONT_PATH,16)
COLORS=palette()


def load_json(path):
    return json.loads(path.read_text())


def fitted(image,size,resample):
    return ImageOps.contain(image,size,method=resample)


def make_comparison(name,stage,image,gt,baseline,adaptive,scores):
    width,height=360,300
    canvas=Image.new('RGB',(width*4,410),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f'{name} | {"Sau DenseCRF" if stage=="crf" else "Trước DenseCRF"}',fill='black',font=FONT)
    titles=['Ảnh gốc','Ground truth',f'Baseline: {scores[0]*100:.2f}%',f'Adaptive: {scores[1]*100:.2f}%']
    images=[image,Image.fromarray(COLORS[gt]),Image.fromarray(COLORS[baseline]),Image.fromarray(COLORS[adaptive])]
    for idx,(title,pic) in enumerate(zip(titles,images)):
        draw.text((idx*width+12,43),title,fill='black',font=FONT)
        view=fitted(pic,(width-16,height),Image.Resampling.LANCZOS if idx==0 else Image.Resampling.NEAREST)
        canvas.paste(view,(idx*width+(width-view.width)//2,76+(height-view.height)//2))
    draw.text((12,382),f'Δ adaptive − baseline: {(scores[1]-scores[0])*100:+.2f} điểm. mIoU ảnh: trung bình IoU các lớp có trong GT.',fill='black',font=SMALL)
    return canvas


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    opts=ap.parse_args(); result=opts.results.resolve()
    runs={n:load_json(result/n/'metrics.json') for n in ['baseline','adaptive']}
    records={n:load_json(result/n/'per_image.json') for n in runs}
    names=list(records['baseline'])
    assert set(names)==set(records['adaptive'])
    assert len(names)==runs['baseline']['metadata']['validation_images']==runs['adaptive']['metadata']['validation_images']
    assert not any(r['metadata']['partial_validation'] for r in runs.values()) or len(names)<1449
    cfg=load_json(result/'baseline/training_config.json')
    dataset=ROOT/cfg['data_folder']
    classes=runs['baseline']['class_names']
    if not runs['baseline']['metadata']['partial_validation']:
        assert set(names)==set((ROOT/cfg['list_folder']/'val.txt').read_text().split())
    (result/'comparisons/raw').mkdir(parents=True,exist_ok=True)
    (result/'comparisons/crf').mkdir(parents=True,exist_ok=True)
    histograms={(mode,stage):np.zeros((21,21),dtype=np.int64) for mode in runs for stage in ['raw','crf']}
    per_image=[]
    for index,name in enumerate(names):
        image=Image.open(dataset/'JPEGImages'/f'{name}.jpg').convert('RGB')
        gt=np.array(Image.open(dataset/'SegmentationClassAug'/f'{name}.png'))
        row={'image':name,'classes':';'.join(classes[c] for c in np.unique(gt) if 0<c<21)}
        for stage in ['raw','crf']:
            scores=[]; masks=[]
            for mode in runs:
                mask=np.array(Image.open(result/mode/stage/'masks'/f'{name}.png'))
                assert mask.shape==gt.shape and mask.dtype==np.uint8
                assert mask.min()>=0 and mask.max()<21
                hist=hist_for(gt,mask)
                histograms[mode,stage]+=hist
                score=image_score(hist)
                assert abs(score-records[mode][name][f'{stage}_miou'])<1e-12
                row[f'{mode}_{stage}_image_miou_percent']=score*100
                scores.append(score);masks.append(mask)
            row[f'{stage}_delta_pp']=(scores[1]-scores[0])*100
            canvas=make_comparison(name,stage,image,gt,*masks,scores)
            canvas.save(result/'comparisons'/stage/f'{name}.png',compress_level=3)
        per_image.append(row)
        if (index+1)%200==0 or index+1==len(names):
            print(f'Comparisons: {index+1}/{len(names)}',flush=True)
    for key,hist in histograms.items():
        mode,stage=key
        assert np.array_equal(hist,np.load(result/mode/f'{stage}_confusion.npy'))
        union=hist.sum(0)+hist.sum(1)-hist.diagonal()
        miou=np.mean(hist.diagonal()[hist.sum(1)>0]/union[hist.sum(1)>0])
        assert abs(miou-runs[mode][stage]['miou'])<1e-12
    with (result/'per_image_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_image[0]))
        writer.writeheader();writer.writerows(per_image)
    with (result/'per_class_comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['class','baseline_raw_iou_percent','adaptive_raw_iou_percent','raw_delta_pp','baseline_crf_iou_percent','adaptive_crf_iou_percent','crf_delta_pp'])
        for idx,cls in enumerate(classes):
            values=[]
            for stage in ['raw','crf']:
                b=runs['baseline'][stage]['iou'][str(idx)]
                a=runs['adaptive'][stage]['iou'][str(idx)]
                values += [None if b is None else b*100,
                           None if a is None else a*100,
                           None if a is None or b is None else (a-b)*100]
            writer.writerow([cls]+values)
    legend=Image.new('RGB',(1000,340),'white');draw=ImageDraw.Draw(legend)
    draw.text((16,8),'Bảng màu VOC: cùng một lớp dùng cùng màu trong GT và hai dự đoán',fill='black',font=FONT)
    for idx,cls in enumerate(classes+['ignore (GT = 255)']):
        x=16+(idx%4)*245;y=54+(idx//4)*46
        color=tuple(COLORS[idx if idx<21 else 255])
        draw.rectangle((x,y,x+28,y+28),fill=color,outline='gray')
        draw.text((x+38,y+4),cls,fill='black',font=SMALL)
    legend.save(result/'legend.png')
    for stage in ['raw','crf']:
        ranked=sorted(per_image,key=lambda r:r[f'{stage}_delta_pp'])
        selected=ranked[:3]+list(reversed(ranked[-3:])) if len(ranked)>=6 else ranked
        overview=Image.new('RGB',(1440,410*len(selected)),'white')
        for idx,row in enumerate(selected):
            with Image.open(result/'comparisons'/stage/f'{row["image"]}.png') as image:
                overview.paste(image,(0,410*idx))
        overview.save(result/f'overview_{stage}.png',compress_level=3)
    summary=load_json(result/'summary.json')
    lines=['# Xem kết quả inference VOC val','',
           'Hai checkpoint 20k được đánh giá cùng scales 1.0, 1.25, 1.5, horizontal flip và công thức tổng hợp của tools/infer_seg_voc.py.',
           'DenseCRF: 10 vòng; pos_xy_std=2, pos_w=3, bi_xy_std=140, bi_rgb_std=1, bi_w=7.','',
           '| Kết quả mIoU (%) | Baseline | Adaptive | Δ adaptive − baseline (điểm) |',
           '|---|---:|---:|---:|']
    for stage,label in [('raw','Trước CRF'),('crf','Sau CRF')]:
        lines.append(f'| {label} | {summary["baseline"][stage]:.4f} | {summary["adaptive"][stage]:.4f} | {summary["delta_adaptive_minus_baseline_pp"][stage]:+.4f} |')
    lines += ['', '## Cách xem ảnh','',
              '1. Trong VS Code/JupyterLab, mở thư mục kết quả này.',
              '2. Mở comparisons/crf/<image_id>.png để xem 4 cột: ảnh gốc → ground truth → baseline → adaptive.',
              '3. Mở comparisons/raw/<image_id>.png để xem cùng ảnh trước CRF.',
              '4. Mở legend.png để xem tên lớp tương ứng từng màu. Vùng xám trong GT là ignore (255), không tính điểm.',
              '5. overview_crf.png / overview_raw.png: 3 ảnh adaptive giảm mạnh nhất, rồi 3 ảnh tăng mạnh nhất theo chênh lệch image mIoU. Đây là các trường hợp cực trị, không đại diện toàn bộ val.',
              '6. Mở per_image_comparison.csv và sắp xếp crf_delta_pp để tìm các ảnh cần kiểm tra; cột classes liệt kê lớp foreground trong GT.','',
              'Điểm trên từng ảnh chỉ trung bình trên các lớp có trong GT ảnh đó. mIoU toàn bộ 1.449 ảnh tính từ confusion matrix cộng dồn, không lấy trung bình điểm từng ảnh.','',
              '## Các file','',
              '- baseline|adaptive/raw|crf/color/*.png: mask màu ở độ phân giải ảnh gốc.',
              '- baseline|adaptive/raw|crf/masks/*.png: mask ID uint8, giá trị 0..20. Mask ID có thể trông đen khi mở bằng trình xem ảnh thông thường; dùng thư mục color hoặc comparisons.',
              '- baseline|adaptive/metrics.json: đầy đủ IoU, precision, recall theo lớp và thông tin checkpoint.',
              '- per_class_comparison.csv: so sánh IoU 21 lớp.',
              '- summary.json: bảng mIoU tổng.',
              '- inference.log, progress.json: nhật ký và trạng thái chạy.','',
              'Đã kiểm tra lại toàn bộ mask: đủ ảnh, đúng kích thước, nhãn 0..20 và confusion matrix khớp các metrics đã ghi.']
    (result/'HUONG_DAN_XEM_ANH.md').write_text('\n'.join(lines)+'\n')
    (result/'validation_checks.json').write_text(json.dumps({
        'images_per_run':len(names), 'comparison_images':2*len(names),
        'mask_shapes_and_class_ids_valid':True,
        'exported_masks_match_saved_confusion_matrices':True,
        'exported_masks_match_saved_miou':True,
        'partial_validation':runs['baseline']['metadata']['partial_validation'],
    },indent=2)+'\n')
    print('Saved comparisons, rankings, legend and viewing guide:',result,flush=True)


if __name__=='__main__':
    main()
