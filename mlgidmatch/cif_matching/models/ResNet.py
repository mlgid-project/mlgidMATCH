import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

class IMGClassifier(nn.Module):
    """ A class for crystal structure matching by using pseudo-image as input """
    def __init__(self, input_dim=14, output_dim=1, res=18):
        super(IMGClassifier, self).__init__()

        if res == 50:
            self.model = resnet50()
            self.model.fc = nn.Linear(in_features=2048, out_features=output_dim, bias=True)
        elif res == 34:
            self.model = resnet34()
            self.model.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)
        elif res == 18:
            self.model = resnet18()
            self.model.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)

        self.model.conv1 = nn.Conv2d(input_dim, 64,
                               kernel_size=(7, 7), stride=(2, 2),
                               bias=False)

    def forward(self, img):
        out = self.model(img)
        return out