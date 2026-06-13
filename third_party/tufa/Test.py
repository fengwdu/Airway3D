from .Config import cfg

from .Prompt import Face_prompt
import third_party.tufa.utils as utils

import torch
import numpy as np
import cv2
import time
from batch_face import RetinaFace

import torchvision.transforms as transforms


def transform_pixel_v2(pt, trans, inverse=False):    # 图像变换函数
    if inverse is False:
        pt = pt @ (trans[:,0:2].T) + trans[:,2]
    else:
        pt = (pt - trans[:,2]) @ np.linalg.inv(trans[:,0:2].T)
    return pt


_retina_cache = None
_tufa_cache = None


def _get_retina():
    global _retina_cache
    if _retina_cache is None:
        _retina_cache = RetinaFace(gpu_id=0, fp16=True)
    return _retina_cache


def _get_tufa():
    global _tufa_cache
    if _tufa_cache is None:
        model = Face_prompt(cfg.MODEL.TYPE)
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        checkpoint = torch.load('./third_party/tufa/Weights/Face_prompt_base.pth')
        model.module.load_state_dict(checkpoint)
        model.eval()
        _tufa_cache = model
    return _tufa_cache


def detect_face_bbox(closed_rgb, open_rgb):
    detector = _get_retina()
    faces = detector([closed_rgb, open_rgb], max_size=1080, threshold=0.95,
                     return_dict=True, batch_size=2)
    return (np.array(faces[0][0]['box'].astype(np.int32)),
            np.array(faces[1][0]['box'].astype(np.int32)))


def get_lmk(rgb_closed, rgb_open, bbox_closed=None, bbox_open=None):
    """TUFA 68 点关键点检测; 支持外部传入 xyxy bbox."""

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = _get_tufa()

    prompt_file = np.load('./third_party/tufa/Prompt/shape_68.npz')['offset'] / 256.0
    prompt_file = torch.from_numpy(prompt_file).float().cuda().unsqueeze(0)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

    image_size = cfg.MODEL.IMG_SIZE
    scale = cfg.ALL.FRACTION

    with torch.no_grad():
        if bbox_closed is None or bbox_open is None:
            t_bbox = time.time()
            bbox_closed, bbox_open = detect_face_bbox(rgb_closed, rgb_open)
            print(f"[Timing] Bbox 检测 (RetinaFace): {time.time() - t_bbox:.2f}s")
        else:
            bbox_closed = np.asarray(bbox_closed, dtype=np.int32)
            bbox_open = np.asarray(bbox_open, dtype=np.int32)

        # RetinaFace 返回 [x1,y1,x2,y2], get_transforms 期望 [x,y,w,h]
        def xyxy_to_xywh(b):
            return np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], dtype=b.dtype)

        bbox_closed_xywh = xyxy_to_xywh(bbox_closed)
        bbox_open_xywh   = xyxy_to_xywh(bbox_open)

        trans_closed = utils.get_transforms(bbox_closed_xywh, scale, 0.0, image_size, shift_factor=[0.0, 0.0])
        trans_open = utils.get_transforms(bbox_open_xywh, scale, 0.0, image_size, shift_factor=[0.0, 0.0])

        crop_closed = cv2.warpAffine(
            rgb_closed, trans_closed,
            (int(image_size), int(image_size)),
            flags=cv2.INTER_LINEAR
        )

        crop_open = cv2.warpAffine(
            rgb_open, trans_open,
            (int(image_size), int(image_size)),
            flags=cv2.INTER_LINEAR
        )

        x_closed = to_tensor(crop_closed).unsqueeze(0).cuda()
        x_open   = to_tensor(crop_open).unsqueeze(0).cuda()

        t_lmk = time.time()
        closed_pred_crop = model(x_closed, prompt_file)[0, -1, :, :].cpu().numpy() * float(image_size)
        open_pred_crop   = model(x_open,   prompt_file)[0, -1, :, :].cpu().numpy() * float(image_size)
        print(f"[Timing] Landmark 预测: {time.time() - t_lmk:.2f}s")

        lmk_closed = transform_pixel_v2(closed_pred_crop, trans_closed, inverse=True)
        lmk_open = transform_pixel_v2(open_pred_crop, trans_open, inverse=True)

    return lmk_closed, lmk_open


def get_lmk_single(rgb, bbox_xywh):
    """单帧 TUFA 68 点关键点检测.

    Args:
        rgb:       (H, W, 3) RGB 图像
        bbox_xywh: array-like [x, y, w, h] — YuNet / OpenCV FaceDetectorYN 格式，
                   与 get_transforms 期望的格式一致，无需额外转换。

    Returns:
        lmk2d: (68, 2) float32，原图坐标系下的关键点像素坐标
    """
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = _get_tufa()

    prompt_file = np.load('./third_party/tufa/Prompt/shape_68.npz')['offset'] / 256.0
    prompt_file = torch.from_numpy(prompt_file).float().cuda().unsqueeze(0)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

    image_size = cfg.MODEL.IMG_SIZE
    scale = cfg.ALL.FRACTION

    # bbox_xywh 来自 YuNet，格式已是 [x, y, w, h]，与 get_transforms 期望一致，直接传入
    bbox = np.asarray(bbox_xywh, dtype=np.int32)

    with torch.no_grad():
        trans = utils.get_transforms(bbox, scale, 0.0, image_size, shift_factor=[0.0, 0.0])

        crop = cv2.warpAffine(
            rgb, trans,
            (int(image_size), int(image_size)),
            flags=cv2.INTER_LINEAR
        )

        x = to_tensor(crop).unsqueeze(0).cuda()

        t_lmk = time.time()
        pred_crop = model(x, prompt_file)[0, -1, :, :].cpu().numpy() * float(image_size)
        print(f"[Timing] Landmark 预测 (single): {time.time() - t_lmk:.2f}s")

        lmk2d = transform_pixel_v2(pred_crop, trans, inverse=True)

    return lmk2d