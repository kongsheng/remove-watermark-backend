from typing import Optional
from PIL import Image
import numpy as np
import cv2

# 依赖：torch + lama-cleaner
try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from lama_cleaner.model_manager import ModelManager  # type: ignore
except Exception:
    ModelManager = None


class LamaNotReady(Exception):
    pass


_manager: Optional[ModelManager] = None


def _ensure_ready():
    if torch is None:
        raise LamaNotReady("未检测到 PyTorch，请先安装（CPU 或 CUDA 版）。")
    if ModelManager is None:
        raise LamaNotReady("未检测到 lama-cleaner，请先执行：pip install lama-cleaner")


def _get_manager() -> ModelManager:
    global _manager
    _ensure_ready()
    if _manager is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 懒加载一次模型，后续复用
        _manager = ModelManager(name='lama', device=device)
    return _manager


from lama_cleaner.schema import Config, HDStrategy  # type: ignore


def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """使用 LaMa 进行图像修复。
    - image: RGB PIL.Image
    - mask: L 模式（黑=保留，白=修复）PIL.Image
    返回: 修复后的 RGB PIL.Image
    """
    manager = _get_manager()

    # 构造最小可用配置（与 lama-cleaner 的 ModelManager.__call__ 签名匹配）
    cfg = Config(
        ldm_steps=16,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=768,
        hd_strategy_resize_limit=1024,
    )

    # 为避免超大图导致内存不足：若最长边>1600，先等比例缩小到1600再推理，最后按掩码区域融合回原图。
    max_edge = max(image.width, image.height)
    target_edge = 1600
    scale = 1.0
    work_img = image
    work_mask = mask
    if max_edge > target_edge:
        scale = target_edge / float(max_edge)
        new_w = int(round(image.width * scale))
        new_h = int(round(image.height * scale))
        work_img = image.resize((new_w, new_h), Image.LANCZOS)
        work_mask = mask.resize((new_w, new_h), Image.NEAREST)

    # 转 numpy 并二值化掩码 + 适度膨胀（扩大修复区域）
    img_np = np.array(work_img)
    mask_np = np.array(work_mask)
    mask_bin = (mask_np > 127).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

    result = manager(img_np, mask_bin, cfg)

    # 统一为 ndarray 以便与原图做逐像素合成（只替换掩码区域）
    if isinstance(result, Image.Image):
        out_np = np.array(result)
    elif isinstance(result, np.ndarray):
        out_np = result
    else:
        raise LamaNotReady("LaMa 返回结果类型异常，请检查 lama-cleaner 版本与依赖。")

    # 若做了下采样，需将输出/掩码按比例映射回原图
    if scale != 1.0:
        # 先把工作域结果 resize 到工作域大小的 out_np（已是），再上采样到原图尺寸供合成
        out_full = cv2.resize(out_np, (image.width, image.height))
        mask_full = cv2.resize(mask_bin, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
        orig_np = np.array(image)
        out_np = out_full
        mask_bin = mask_full
    else:
        orig_np = np.array(image)

    # 掩码转 3 通道布尔，只在掩码区域替换像素，非掩码区域保留原图颜色，避免整图色偏
    mask_bool = (mask_bin > 0)
    if mask_bool.ndim == 2:
        mask_bool = mask_bool[..., None]
    if out_np.ndim == 2:
        out_np = cv2.cvtColor(out_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if orig_np.ndim == 2:
        orig_np = cv2.cvtColor(orig_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Feather 边缘，缓解多色背景的接缝/色差（将掩码平滑为 0~1 alpha）
    alpha = (mask_bool.astype(np.float32))
    alpha = cv2.GaussianBlur(alpha, (11, 11), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 小区域采用 Poisson 混合提升色彩一致性（LOGO/角落小水印场景）
    area = int(mask_bin.sum() // 255)
    h, w = orig_np.shape[:2]
    if area < (h * w * 0.02):  # 小于 2% 面积使用 Poisson
        mask3 = (mask_bin > 0).astype(np.uint8) * 255
        if mask3.ndim == 2:
            mask3 = cv2.merge([mask3, mask3, mask3])
        ys, xs = np.where(mask_bin > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            try:
                blended_cv = cv2.seamlessClone(out_np.astype(np.uint8), orig_np.astype(np.uint8), mask3, (cx, cy), cv2.NORMAL_CLONE)
                return Image.fromarray(blended_cv)
            except Exception:
                pass  # 回退到 alpha 混合

    blended = (alpha * out_np + (1 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)
