from .conversion import convert
from torch.utils.data import DataLoader

def setup(stage, config, model):
    print("setup called!")
    if stage in ['test', 'predict'] and stage not in ['fit', 'validate']:

        train_dataset = config.train_dataset(transform = config.preprocessors.train)
        train_dataloader = DataLoader(
            train_dataset, batch_size= config.batch_size,
            shuffle=True, drop_last=False, num_workers=4, pin_memory=True
        )
        snn_model, ported_ann_model, sample = convert(
            model,
            config.neuronal_dynamics,
            config.dynamics_type,
            config.default_simulation_length,
            train_dataloader,
            config.max_activation_scale_iterations,
            config.scale_relu_with_max_activation,
            None
        )
        reference_ann_model, device = ported_ann_model, None
        return snn_model, ported_ann_model, model, device
    return model
