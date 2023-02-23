# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
# https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py

from . import convert


def load_cameras_all(sparse_dir):
    """
    """
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "images.bin").exists():
        ims = read_images_binary(sparse_dir / "images.bin")
    else:
        ims = read_images_text(sparse_dir / "images.txt")
    if (sparse_dir / "cameras.bin").exists():
        cams = read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        cams = read_cameras_text(sparse_dir / "cameras.txt")

    ims = {im.name: im for key, im in ims.items()}

    Ks = []
    Rs = []
    ts = []
    names = []
    heights, widths = [], []
    for im_name in ims.keys():
        im = ims[im_name]
        camera_id = im.camera_id
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cams[camera_id].params
        Ks.append(K)
        Rs.append(rotm_from_quat(im.qvec))
        ts.append(im.tvec)
        heights.append(cams[camera_id].height)
        widths.append(cams[camera_id].width)
        names.append(im_name)

    Ks = np.array(Ks)
    Rs = np.array(Rs)
    ts = np.array(ts)
    heights = np.array(heights)
    widths = np.array(widths)

    return ims, Ks, Rs, ts, heights, widths, names


def main_bin_to_txt(args):
    """
    """
    print("bin-to-txt")
    sparse_dir = Path(args.sparse_dir)
    print(sparse_dir)

    cam_bin = sparse_dir / "cameras.bin"
    cam_txt = sparse_dir / "cameras.txt"
    if cam_bin.exists():
        print(f"convert {cam_bin} to {cam_txt}")
        cameras = read_cameras_binary(str(cam_bin))
        write_cameras_text(cam_txt, cameras)

    img_bin = sparse_dir / "images.bin"
    img_txt = sparse_dir / "images.txt"
    if img_bin.exists():
        print(f"convert {img_bin} to {img_txt}")
        images = read_images_binary(str(img_bin))
        write_images_text(img_txt, images)

    pts_bin = sparse_dir / "points3D.bin"
    pts_txt = sparse_dir / "points3D.txt"
    if pts_bin.exists():
        print(f"convert {pts_bin} to {pts_txt}")
        points3d = read_points3d_binary(str(pts_bin))
        write_points3D_text(pts_txt, points3d)


def read_colmap_to_Ks_Ts_names(data_dir):
    """
    Args:
        data_dir: where cameras.bin, images.bin, points3D.bin are located

    Returns:
        tuple(Ks, Ts)
        Ks: list of camera intrinsics, Nx3x3.
        Ts: list of camera extrinsics, Nx4x4.

    We assume the original images are sorted by their file names. The returned
    Ks and Ts are sorted by the original image file names.
    """

    images_dict = read_images_binary(data_dir / "images.bin")
    cameras_dict = read_cameras_binary(data_dir / "cameras.bin")

    image_name_to_image_key = {}
    for image_key, image_obj in images_dict.items():
        image_name_to_image_key[image_obj.name] = image_key

    # We assume the original images are sorted by their file names.
    image_names = sorted(list(image_name_to_image_key.keys()))
    Ks = []
    Ts = []
    for image_name in image_names:
        image_key = image_name_to_image_key[image_name]
        image_obj = images_dict[image_key]
        R = image_obj.qvec2rotmat()
        t = image_obj.tvec
        T = convert.R_t_to_T(R, t)
        Ts.append(T)

        camera = cameras_dict[image_obj.camera_id]
        K = np.eye(3).astype(np.float64)

        # https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
        if camera.model == "SIMPLE_PINHOLE":
            # f, cx, cy
            K[0, 0] = camera.params[0]
            K[1, 1] = camera.params[0]
            K[0, 2] = camera.params[1]
            K[1, 2] = camera.params[2]
        elif camera.model == "PINHOLE":
            # fx, fy, cx, cy
            K[0, 0] = camera.params[0]
            K[1, 1] = camera.params[1]
            K[0, 2] = camera.params[2]
            K[1, 2] = camera.params[3]
        elif camera.model == "SIMPLE_RADIAL":
            # f, cx, cy, k
            k = camera.params[3]
            # print(f"Warning: SIMPLE_RADIAL camera distortion {k} ignored.")
            K[0, 0] = camera.params[0]
            K[1, 1] = camera.params[0]
            K[0, 2] = camera.params[1]
            K[1, 2] = camera.params[2]
        else:
            import ipdb
            ipdb.set_trace()
            raise ValueError(f"Unknown camera model: {camera.model}")

        Ks.append(K)

    assert len(Ks) == len(Ts) == len(image_names)

    return np.array(Ks), np.array(Ts), image_names
