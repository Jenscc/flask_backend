import torch
from model import UNet
import utils
import torchvision.transforms as T

localizationWeights = r'../checkpoints/MBM/epoch_10.pth.tar'


def detect(srcImgPath):
    model = UNet(2).to(utils.device())
    model.load_state_dict(torch.load(localizationWeights, map_location=utils.device())['state_dict'])
    model.eval()

    image = utils.read_image(srcImgPath)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    x = transform(image).unsqueeze(0)
    points = None
    with torch.no_grad():
        x = x.to(utils.device())
        pred = utils.ensure_array(model(x)).squeeze(0)
        points = utils.extract_points_from_direction_field_map(pred, lambda1=0.7, step=10)