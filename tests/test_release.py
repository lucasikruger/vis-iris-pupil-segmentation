from pathlib import Path
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from src.models import build_model
from src.splits import partition
from src.losses import SegmentationLoss
from src.mask_formats import get_converter

def test_mask_palette():
    bgr=np.array([[[0,0,255],[0,255,0],[255,0,0],[255,255,0]]],dtype=np.uint8)
    assert get_converter('mobius')(bgr).tolist()==[[0,1,2,2]]

def test_splits():
    df=pd.read_csv('data/partitions.csv')
    assert len(df)==16898
    assert df.split.value_counts().to_dict()=={'train':11605,'test':3542,'val':1751}
    for fold in ['mobius','polyu_zoomed','ubiris_v2_with_pupils_zoomed','fei_manual','miche','all']:
        x=partition(df,fold)
        a=set(x[x.split=='train'].subject_id);b=set(x[x.split=='val'].subject_id)
        assert not a&b
        assert (x[x.dataset.isin(['closed_eyes_bg','white_eyes_bg'])].split=='train').all()
        if fold!='all':assert set(x[x.split=='test'].dataset)=={fold}
    for field in ['rel_image_path','rel_mask_path']:
        assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in df[field])

def test_loss_gradient():
    x=torch.randn(2,3,96,96,requires_grad=True);y=torch.zeros((2,96,96),dtype=torch.long)
    y[:,30:60,30:60]=1;y[:,40:50,40:50]=2
    loss=SegmentationLoss()(x,y);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_checkpoint_onnx_parity():
    torch.set_num_threads(4);torch.manual_seed(3)
    model=build_model();ckpt=torch.load('weights/model.pt',map_location='cpu',weights_only=True)
    assert ckpt['epoch']==99 and ckpt['uses_ema'] and ckpt['test_dataset']=='mobius'
    model.load_state_dict(ckpt['model_state']);model.eval()
    assert sum(p.numel() for p in model.parameters())==323981
    session=ort.InferenceSession('demos/web/public/models/res_unet_in_b2__96.onnx',providers=['CPUExecutionProvider'])
    for x in [torch.rand(1,3,96,96),torch.zeros(1,3,96,96)]:
        with torch.no_grad():expected=model(x).numpy()
        actual=session.run(None,{session.get_inputs()[0].name:x.numpy()})[0]
        if x.count_nonzero():
            np.testing.assert_allclose(expected,actual,rtol=0.002,atol=0.002)
        else:
            # InstanceNorm on a constant input amplifies floating-point differences.
            assert np.max(abs(expected-actual)) < 0.03
            np.testing.assert_array_equal(expected.argmax(1),actual.argmax(1))


def test_public_occlusion_fallback_and_optional_assets(monkeypatch):
    import src.augmentations as aug
    image = np.full((96,96,3),128,dtype=np.uint8)
    mask = np.zeros((96,96),dtype=np.int64)
    mask[25:70,25:70] = 1
    mask[40:55,40:55] = 2
    monkeypatch.setattr(aug, '_OCC_CACHE', [])
    np.random.seed(42)
    result, labels = aug.aug_occlusion(image,mask)
    assert result.shape == image.shape and labels.shape == mask.shape
    assert set(np.unique(labels)) <= {0,1,2}
    assert np.any(labels != mask)  # Procedural occlusion is active, not a no-op.
    asset = np.full((32,32,4),255,dtype=np.uint8)
    rules = dict(aug._OCC_RULES[aug._occ_cat('glasses_lens.png')])
    monkeypatch.setattr(aug,'_OCC_CACHE',[('glasses_lens.png',asset,rules)])
    def no_fallback(*args,**kwargs):
        raise AssertionError('Assets presentes: no debe usarse el fallback')
    monkeypatch.setattr(aug,'aug_eyelid_occlusion_v3',no_fallback)
    result, labels = aug.aug_occlusion(image,mask)
    assert result.shape == image.shape


def test_recorded_crop_is_half_open_and_checks_dimensions():
    from src.crop_metadata import crop_from_record, relative_path
    import pytest
    image=np.arange(8*10*3,dtype=np.uint8).reshape(8,10,3)
    record={'source_size':[10,8],'bbox_xyxy':[2,1,7,6],'output_size':[5,5],'resize':'none'}
    np.testing.assert_array_equal(crop_from_record(image,record),image[1:6,2:7])
    with pytest.raises(ValueError):crop_from_record(image,{**record,'source_size':[9,8]})
    with pytest.raises(ValueError):crop_from_record(image,{**record,'bbox_xyxy':[-1,1,7,6]})
    with pytest.raises(ValueError):relative_path('data','../escape')


def test_webcam_record_preserves_detector_and_saved_frames(tmp_path):
    import json
    from src.webcam import RecorteConPadding
    (tmp_path/'annotations').mkdir()
    r={'bbox_xyxy':[10,19,70,79],'raw_crop_bbox_xyxy':[10,20,71,81],
       'raw_contour_crop_xy':[[10,20],[40,20],[40,30],[10,30]]}
    (tmp_path/'annotations/webcam_boxes.json').write_text(json.dumps({
        'schema_version':1,'coordinate_system':'pixel_xyxy_half_open','records':{'eye':r}}))
    dataset=RecorteConPadding(pd.DataFrame(),0.12,tmp_path)
    assert dataset.bb['eye']['crop_bbox_in_orig']==[10,19,70,79]
    assert dataset.crudo['eye']['y']==20
    assert dataset.caja('eye')==(55,63,317)


def test_postprocess_never_creates_absent_pupil():
    import numpy as np
    from src.postprocess import postprocess
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:27, 5:27] = 1
    mask[10:20, 10:20] = 0
    result = postprocess(mask)
    assert not (result == 2).any()
    assert result[15, 15] == 1
    mask[12:18, 12:18] = 2
    mask[0, 0] = 2
    result = postprocess(mask)
    assert result[15, 15] == 2
    assert result[0, 0] == 0
