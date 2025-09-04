import torch
from torch.utils.data import DataLoader
from cifar10.models.models import list_models, load_model
from cifar10.dataset import config as cifar10_config
from cifar10.preprocess import Cutout, CIFAR10Policy

from torch.nn import *
from torch.nn.init import normal_, xavier_normal_
from torchvision.transforms import *

from snn_signgd import convert

DEV = "cpu"

transforms_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    CIFAR10Policy(),
    ToTensor(),
    Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
    Cutout(n_holes=1, length=16)
])
transforms_test = Compose([
    ToTensor(),
    Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])
if __name__ == '__main__':
    print(list_models(verbose=True))

    batch_size = 256
    timestamps = [1, 2, 4, 8, 16, 32, 64]

    # CHANGE THIS LINE TO TEST DIFFERENT DYNAMICS
    dynamics_type = 'signgd' # 'subgradient'
    match dynamics_type:
        case 'signgd':
            from signgd_dynamics import config
        case 'subgradient':
            from subgradient_dynamics import config
        case _:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}, must be 'signgd' or 'subgradient'")

    ann = load_model(
        category_name = 'resnet',
        vgg_default = 'VGG16',
        index = 0
    ).to(DEV)

    ann.eval()

    # Interesting initialization
    def init_weights(m):
        if type(m) == Linear or type(m) == Conv2d:
            xavier_normal_(m.weight, gain=1.0)
            normal_(m.bias, mean=1.0, std=1)
    ann.apply(init_weights)

    d_tr = cifar10_config.train_dataset(transform = transforms_train)
    l_tr = DataLoader(
        d_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    d_te = cifar10_config.test_dataset(transform = transforms_test)
    l_te = DataLoader(
        d_te,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        snn, ann_model_ported, convert_sample = convert(
            ann,
            config.neuronal_dynamics,
            dynamics_type,
            64,
            l_tr,
            10,
            True,
            None
        )
        print(snn)

        sample, _ = next(iter(l_te))
        x = sample.to(DEV)

        ann_output = ann(x)
        print("ANN output:", ann_output.shape, "\n", ann_output)

        snn_output, outputs_timestamp = snn(x, timestamps)
        print("SNN output:", snn_output.shape, "\n" ,snn_output)
