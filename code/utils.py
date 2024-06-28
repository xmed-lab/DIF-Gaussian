import yaml
import numpy as np
import SimpleITK as sitk
from easydict import EasyDict



def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda()
    return item


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
        origin *= 1000
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing, origin


def sitk_save(path, image, spacing=None, origin=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    if origin is not None:
        out.SetOrigin(origin.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)
