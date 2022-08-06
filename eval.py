import numpy as np
from PIL import Image
from spade.model import Pix2PixModel
from spade.dataset import get_transform
from torchvision.transforms import ToPILImage

colorMap = [
    {"color": (56, 79, 131), "id": 154, "label": "sea"},
    {"color": (239, 239, 239), "id": 105, "label": "cloud"},
    {"color": (93, 110, 50), "id": 96, "label": "bush"},
    {"color": (183, 210, 78), "id": 123, "label": "grass"},
    {"color": (60, 59, 75), "id": 134, "label": "mountain"},
    {"color": (117, 158, 223), "id": 156, "label": "sky"},
    {"color": (250, 250, 250), "id": 158, "label": "snow"},
    {"color": (53, 38, 19), "id": 168, "label": "tree"},
    {"color": (230, 112, 182), "id": 118, "label": "flower"},
    {"color": (152, 126, 106), "id": 148, "label": "road"}
]

colors = [key['color'] for key in colorMap]
id_list = [key['id'] for key in colorMap]


def semantic(img):
    print("semantic", type(img))
    h, w = img.size
    imrgb = img.convert("RGB")
    pix = list(imrgb.getdata())
    mask = [id_list[colors.index(i)] if i in colors else 156 for i in pix]
    return np.array(mask).reshape(h, w)


def evaluate(labelmap):
    opt = {
        'label_nc': 182,
        'crop_size': 512,
        'load_size': 512,
        'aspect_ratio': 1.0,
        'isTrain': False,
        'checkpoints_dir': 'app',
        'which_epoch': 'latest',
        'use_gpu': False
    }
    model = Pix2PixModel(opt)
    model.eval()
    image = Image.fromarray(np.array(labelmap).astype(np.uint8))
    transform_label = get_transform(opt, method=Image.NEAREST, normalize=False)

    label_tensor = transform_label(image) * 255.0
    label_tensor[label_tensor == 255] = opt['label_nc']
    print("label_tensor:", label_tensor.shape)

    transform_image = get_transform(opt)
    image_tensor = transform_image(Image.new('RGB', (500, 500)))

    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': image_tensor.unsqueeze(0)
    }
    generated = model(data, mode='inference')
    print("generated_image:", generated.shape)

    return generated


def to_image(generated):
    to_img = ToPILImage()
    normalized_img = ((generated.reshape([3, 512, 512]) + 1) / 2.0) * 255.0
    return to_img(normalized_img.byte().cpu())
