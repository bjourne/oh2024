import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cifar10.models.models import list_models, load_model
from cifar10.dataset import config as cifar10_config
from cifar10.preprocess import Cutout, CIFAR10Policy
from torch.nn import *
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

    ann_model = load_model(
        category_name = 'resnet',
        vgg_default = 'VGG16',
        index = 0
    ).to(DEV)

    ann_model.eval()

    def init_weights(m):
        if type(m) == Linear or type(m) == Conv2d:
            torch.nn.init.xavier_normal_(m.weight,gain=1.0)
            torch.nn.init.normal_(m.bias,mean=1.0,std=1)

    ann_model.apply(init_weights)

    train_dataset = cifar10_config.train_dataset(transform = transforms_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    test_dataset = cifar10_config.test_dataset(transform = transforms_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        snn_model, ann_model_ported, convert_sample = convert(
            ann_model,
            neuronal_dynamics = config.neuronal_dynamics,
            dynamics_type = dynamics_type,
            default_simulation_length = 64,
            activation_scale_dataloader = train_dataloader,
            max_activation_scale_iterations = 10,
            scale_relu_with_max_activation = True
        )

        sample, _ = next(iter(test_dataloader))
        x = sample.to(DEV)

        ann_output = ann_model(x)
        print("ANN output:", ann_output.shape, "\n", ann_output)

        snn_output, outputs_timestamp = snn_model(x, timestamps)
        print("SNN output:", snn_output.shape, "\n" ,snn_output)
