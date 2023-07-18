import numpy as np

from . import image
from . import io
from . import sanity


def image_psnr(im_pd, im_gt, im_mask=None):
    """
    Computes PSNR given images in numpy arrays.

    Args:
        im_pd: numpy array, (h, w, 3), float32, range [0..1].
        im_gt: numpy array, (h, w, 3), float32, range [0..1].
        im_mask: numpy array, (h, w), float32, range [0..1].
            Value > 0.5 means foreground. None means all foreground.

    Returns:
        PSNR value in float.
    """
    if im_mask is None:
        h, w = im_pd.shape[:2]
        im_mask = np.ones((h, w), dtype=np.float32)
    _check_inputs(im_pd, im_gt, im_mask)

    from skimage.metrics import peak_signal_noise_ratio as psnr

    im_mask = im_mask[:, :, None]  # (h, w) -> (h, w, 1)
    pr = im_pd[im_mask[:, :, 0] > 0.5].ravel()
    gt = im_gt[im_mask[:, :, 0] > 0.5].ravel()
    assert pr.dtype == gt.dtype and pr.dtype == np.float32
    ans = psnr(gt, pr)
    return float(ans)


def image_ssim(im_pd, im_gt, im_mask=None):
    """
    Computes SSIM given images in numpy arrays.

    Args:
        im_pd: numpy array, (h, w, 3), float32, range [0..1].
        im_gt: numpy array, (h, w, 3), float32, range [0..1].
        im_mask: numpy array, (h, w), float32, range [0..1].
            Value > 0.5 means foreground. None means all foreground.

    Returns:
        SSIM value in float.
    """
    if im_mask is None:
        h, w = im_pd.shape[:2]
        im_mask = np.ones((h, w), dtype=np.float32)
    _check_inputs(im_pd, im_gt, im_mask)

    from skimage.metrics import structural_similarity as ssim

    im_mask = im_mask[:, :, None]  # (h, w) -> (h, w, 1)
    pr = im_pd * im_mask
    gt = im_gt * im_mask
    assert pr.dtype == gt.dtype and pr.dtype == np.float32
    mean, S = ssim(pr, gt, channel_axis=-1, full=True)
    return float(S[im_mask[:, :, 0] > 0.5].mean())


def image_lpips(im_pd, im_gt, im_mask=None):
    """
    Computes LPIPS given images in numpy arrays.

    Args:
        im_pd: numpy array, (h, w, 3), float32, range [0..1].
        im_gt: numpy array, (h, w, 3), float32, range [0..1].
        im_mask: numpy array, (h, w), float32, range [0..1].
            Value > 0.5 means foreground. None means all foreground.

    Returns:
        LPIPS value in float.
    """
    if im_mask is None:
        h, w = im_pd.shape[:2]
        im_mask = np.ones((h, w), dtype=np.float32)
    _check_inputs(im_pd, im_gt, im_mask)

    import torch
    import lpips

    im_mask = im_mask[:, :, None]  # (h, w) -> (h, w, 1)
    pr = im_mask * (im_pd * 2 - 1)
    gt = im_mask * (im_gt * 2 - 1)
    pr = pr.transpose(2, 0, 1)[None, ...]
    gt = gt.transpose(2, 0, 1)[None, ...]

    if "loss_fn" in image_lpips.static_vars:
        loss_fn = image_lpips.static_vars["loss_fn"]
    else:
        loss_fn = lpips.LPIPS(net="alex")
        image_lpips.static_vars["loss_fn"] = loss_fn

    ans = (
        loss_fn.forward(torch.from_numpy(pr), torch.from_numpy(gt))
        .cpu()
        .detach()
        .numpy()
    )
    return float(ans)


image_lpips.static_vars = {}


def image_psnr_with_paths(im_pd_path, im_gt_path, im_mask_path=None):
    """
    Args:
        im_pd_path: Path to the rendered RGB image. The image will be resized to
            the same (h, w) as im_gt.
        im_gt_path: Path to the ground truth RGB image.
        im_mask_path: Path to the mask image. The mask will be resized to the
            same (h, w) as im_gt.

    Returns:
        PSNR value in float.
    """
    im_pd, im_gt, im_mask = load_im_pd_im_gt_im_mask_for_eval(
        im_pd_path,
        im_gt_path,
        im_mask_path,
        alpha_mode="white",
    )
    return image_psnr(im_pd, im_gt, im_mask)


def image_ssim_with_paths(im_pd_path, im_gt_path, im_mask_path=None):
    """
    Args:
        im_pd_path: Path to the rendered RGB image. The image will be resized to
            the same (h, w) as im_gt.
        im_gt_path: Path to the ground truth RGB image.
        im_mask_path: Path to the mask image. The mask will be resized to the
            same (h, w) as im_gt.

    Returns:
        SSIM value in float.
    """
    im_pd, im_gt, im_mask = load_im_pd_im_gt_im_mask_for_eval(
        im_pd_path,
        im_gt_path,
        im_mask_path,
        alpha_mode="white",
    )
    return image_ssim(im_pd, im_gt, im_mask)


def image_lpips_with_paths(im_pd_path, im_gt_path, im_mask_path=None):
    """
    Args:
        im_pd_path: Path to the rendered RGB image. The image will be resized to
            the same (h, w) as im_gt.
        im_gt_path: Path to the ground truth RGB image.
        im_mask_path: Path to the mask image. The mask will be resized to the
            same (h, w) as im_gt.
    """
    im_pd, im_gt, im_mask = load_im_pd_im_gt_im_mask_for_eval(
        im_pd_path,
        im_gt_path,
        im_mask_path,
        alpha_mode="white",
    )
    return image_lpips(im_pd, im_gt, im_mask)


def load_im_pd_im_gt_im_mask_for_eval(
    im_pd_path, im_gt_path, im_mask_path=None, alpha_mode="white"
):
    """
    Load prediction, ground truth, and mask images for image metric evaluation.

    Args:
        im_pd_path: Path to the rendered image.
        im_gt_path: Path to the ground truth RGB or RGBA image.
        im_mask_path: Path to the mask image. The mask will be resized to the
            same (h, w) as im_gt.
        alpha_mode: The mode on how to handle the alpha channel. Currently only
            "white" is supported.
            - "white": If im_gt contains alpha channel, im_gt will be converted
                       to RGB, the background will be rendered as white, the
                       alpha channel will be then ignored.
            - "keep" : If im_gt contains alpha channel, the alpha channel will
                       be used as mask. This mask can be overwritten by
                       im_mask_path if im_mask_path is not None.
                       (This option is not implemented yet.)

    Returns:
        im_pd: (h, w, 3), float32, value in [0, 1].
        im_gt: (h, w, 3), float32, value in [0, 1].
        im_mask: (h, w), float32, value only 0 or 1. Even if im_mask_path is
            None, im_mask will be returned as all 1s.
    """
    if alpha_mode != "white":
        raise NotImplementedError('Currently only alpha_mode="white" is supported.')

    # Prepare im_gt.
    # (h, w, 3) or (h, w, 4), float32.
    # If (h, w, 4), the alpha channel will be separated.
    im_gt = io.imread(im_gt_path, alpha_mode=alpha_mode)
    if im_gt.shape[2] == 4:
        im_gt_alpha = im_gt[:, :, 3]
        im_gt = im_gt[:, :, :3]
    else:
        im_gt_alpha = None

    # Prepare im_pd.
    # (h, w, 3), float32.
    im_pd = io.imread(im_pd_path)

    # Resize gt and pd to smaller wh.
    gt_w, gt_h = im_gt.shape[1], im_gt.shape[0]
    pd_w, pd_h = im_pd.shape[1], im_pd.shape[0]
    min_wh = min(gt_w, pd_w), min(gt_h, pd_h)
    im_gt = image.resize(im_gt, shape_wh=min_wh)
    if im_gt_alpha is not None:
        im_gt_alpha = image.resize(im_gt_alpha, shape_wh=min_wh)
    im_pd = image.resize(im_pd, shape_wh=min_wh)

    # Prepare im_mask.
    # (h, w), float32, value only 0 or 1.
    if im_mask_path is None:
        if im_gt_alpha is None:
            im_mask = np.ones((min_wh[1], min_wh[0]), dtype=np.float32)
        else:
            im_mask = (im_gt_alpha > 0.5).astype(np.float32)
    else:
        im_mask = io.imread(im_mask_path, alpha_mode="ignore")
        im_mask = image.resize(im_mask, shape_wh=min_wh)
        if im_mask.ndim == 3:
            im_mask = im_mask[:, :, 0]
        im_mask = (im_mask > 0.5).astype(np.float32)

    return im_pd, im_gt, im_mask


def _check_inputs(im_pd, im_gt, im_mask):
    # Instance type.
    sanity.assert_numpy(im_pd, name="im_pd")
    sanity.assert_numpy(im_gt, name="im_gt")
    sanity.assert_numpy(im_mask, name="im_mask")

    # Dtype.
    if im_pd.dtype != np.float32:
        raise ValueError("im_pd must be float32")
    if im_gt.dtype != np.float32:
        raise ValueError("im_gt must be float32")
    if im_mask.dtype != np.float32:
        raise ValueError("im_mask must be float32")

    # Shape.
    sanity.assert_shape(im_pd, (None, None, 3), name="im_pd")
    sanity.assert_shape(im_gt, (None, None, 3), name="im_gt")
    sanity.assert_shape(im_mask, (None, None), name="im_mask")
    if im_pd.shape != im_gt.shape:
        raise ValueError("im_pd and im_gt must have same shape")
    if im_pd.shape[:2] != im_mask.shape:
        raise ValueError("im_pd and im_mask must have same (h, w)")

    # Range.
    if im_pd.max() > 1.0 or im_pd.min() < 0.0:
        raise ValueError("im_pd must be in range [0..1]")
    if im_gt.max() > 1.0 or im_gt.min() < 0.0:
        raise ValueError("im_gt must be in range [0..1]")
    if im_mask.max() > 1.0 or im_mask.min() < 0.0:
        raise ValueError("im_mask must be in range [0..1]")
