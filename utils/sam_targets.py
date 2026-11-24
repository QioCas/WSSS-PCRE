"""Class-agnostic SAM masks acquire VOC classes only through teacher/image tags."""
import numpy as np
from scipy import ndimage as ndi


def choose_mask(masks,scores,reference,core,plan):
    accepted=[];reference_area=int(reference.sum());core_area=int(core.sum())
    if not reference_area or not core_area:return None
    for index,(mask,score) in enumerate(zip(masks,scores)):
        mask=mask.astype(bool);area=int(mask.sum())
        if not area or not np.isfinite(score):continue
        iou=float((mask&reference).sum()/max(1,(mask|reference).sum()))
        recall=float((mask&core).sum()/core_area);ratio=area/reference_area
        if (score>=plan['sam_min_score'] and iou>=plan['sam_min_teacher_iou']
            and recall>=plan['sam_min_core_recall']
            and plan['sam_min_area_ratio']<=ratio<=plan['sam_max_area_ratio']):
            accepted.append((.5*iou+.5*float(score),index,{'teacher_iou':iou,'core_recall':recall,'sam_predicted_quality':float(score)}))
    if not accepted:return None
    _,index,metadata=max(accepted,key=lambda item:item[0])
    return masks[index].astype(bool),metadata


def compose_targets(base,proposals,plan):
    output=base.copy();uncertain=np.zeros_like(base,dtype=bool);background=np.zeros_like(uncertain);foreground={}
    radius=plan['sam_uncertain_radius'];band=plan['sam_boundary_band']
    for proposal in proposals:
        c=proposal['class_id'];mask=proposal['mask'];reference=proposal['reference']
        roi=ndi.binary_dilation(reference,iterations=band)
        allowed=(base==0)|(base==255)|(base==c)
        protected=ndi.binary_erosion(base==c,iterations=radius)
        inner=ndi.binary_erosion(mask,iterations=radius);outer=ndi.binary_dilation(mask,iterations=radius)
        foreground[c]=foreground.get(c,np.zeros_like(uncertain))|(inner&roi&allowed)
        uncertain|=(outer&~inner)&roi&allowed&~protected
        background|=reference&~outer&allowed&~protected
    output[uncertain]=255;output[background]=0
    if foreground:
        ids=sorted(foreground);stack=np.stack([foreground[c] for c in ids]);count=stack.sum(0)
        selected=np.asarray(ids,dtype=np.uint8)[stack.argmax(0)]
        output[count==1]=selected[count==1];output[count>1]=255
    reassigned=(base>0)&(base<21)&(output>0)&(output<21)&(output!=base)
    if reassigned.any():raise AssertionError('SAM must not reassign a reliable foreground class')
    stats={'changed_pixels':int((output!=base).sum()),
           'added_foreground':int((((base==0)|(base==255))&(output>0)&(output<21)).sum()),
           'trimmed_foreground':int(((base>0)&(base<21)&(output==0)).sum()),
           'new_ignore':int(((base!=255)&(output==255)).sum()),'foreground_class_reassignments':0}
    return output,stats


def propose_from_sam(predictor,reliable,reference,image_tags,plan):
    proposals=[];attempted=0
    h,w=reference.shape
    for c in (np.flatnonzero(image_tags>0)+1):
        components,n=ndi.label(reference==c)
        areas=np.bincount(components.ravel());areas[0]=0
        selected=np.argsort(areas)[::-1][:plan['sam_max_components']]
        for index in selected:
            if index==0 or areas[index]<max(32,plan['sam_min_component_fraction']*h*w):continue
            region=components==index;core=region&(reliable==c)
            if core.sum()<16:continue
            y,x=np.unravel_index(ndi.distance_transform_edt(core).argmax(),core.shape)
            yy,xx=np.where(region);pad_x=max(8,int((xx.max()-xx.min()+1)*.1));pad_y=max(8,int((yy.max()-yy.min()+1)*.1))
            box=np.array([max(0,xx.min()-pad_x),max(0,yy.min()-pad_y),min(w-1,xx.max()+pad_x),min(h-1,yy.max()+pad_y)],dtype=np.float32)
            points=[[x,y]];labels=[1]
            other=(reliable>0)&(reliable<21)&(reliable!=c)
            in_box=np.zeros_like(other);in_box[int(box[1]):int(box[3])+1,int(box[0]):int(box[2])+1]=True
            negatives=other&in_box
            if negatives.any():
                distance=ndi.distance_transform_edt(~region);ny,nx=np.where(negatives);j=np.argmin(distance[ny,nx]);points.append([nx[j],ny[j]]);labels.append(0)
            masks,scores,_=predictor.predict(point_coords=np.asarray(points,dtype=np.float32),point_labels=np.asarray(labels),box=box,multimask_output=True)
            attempted+=1;choice=choose_mask(masks,scores,region,core,plan)
            if choice:
                mask,metadata=choice;proposals.append({'class_id':int(c),'mask':mask,'reference':region,**metadata})
    return proposals,attempted
