import os
import torch
import argparse

from agearts.data import lock_deterministic, get_train_aug_preproc, get_valid_aug_preproc
from agearts.utils import get_param_from_config, object_from_dict

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
config_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../configs"))

parser = argparse.ArgumentParser()
parser.add_argument('--config', action="store", dest="config",
                    default=os.path.join(config_dir, "train_regression.yaml"),
                    type=str)

args = parser.parse_args()


def main(train_config: dict):
    TRAIN_PATH = train_config.train_path
    VALID_PATH = train_config.valid_path
    TEST_PATH = train_config.test_path

    ROOT = train_config.root
    updates_per_epoch = train_config.updates_per_epoch

    DEVICE = torch.device(train_config.device)

    ROOT_TO_SAVE_MODEL = train_config.root_to_save_model
    os.makedirs(ROOT_TO_SAVE_MODEL, exist_ok=True)

    lock_deterministic(train_config.seed)

    model = object_from_dict(train_config.model, feature_extracting=True).to(DEVICE)

    PREPROCESSING_FN = model.get_preprocess_fn()

    train_dataset = object_from_dict(train_config.dataset, csv_path=TRAIN_PATH, root_to_data=ROOT,
                                     augmentation=get_train_aug_preproc(PREPROCESSING_FN),
                                     mode='train',
                                     length=updates_per_epoch * train_config.train_dataloader.batch_size)
    train_loader = train_dataset.get_dataloader(**train_config.train_dataloader, shuffle=True)

    valid_dataset = object_from_dict(train_config.dataset, csv_path=VALID_PATH, root_to_data=ROOT,
                                     augmentation=get_valid_aug_preproc(PREPROCESSING_FN),
                                     mode='val')
    valid_loader = valid_dataset.get_dataloader(**train_config.valid_dataloader, shuffle=False)

    test_dataset = object_from_dict(train_config.dataset, csv_path=TEST_PATH, root_to_data=ROOT,
                                     augmentation=get_valid_aug_preproc(PREPROCESSING_FN),
                                     mode='val')
    test_loader = test_dataset.get_dataloader(**train_config.valid_dataloader, shuffle=False)

    optimizer = object_from_dict(train_config.optimizer,
                                 params=filter(lambda x: x.requires_grad, model.get_fc_parameters()))

    criterion = object_from_dict(train_config.criterion)

    scheduler = object_from_dict(train_config.scheduler1, optimizer=optimizer)

    trainer = object_from_dict(
        train_config.trainer1,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        validloader=valid_loader,
        testloader=test_loader,
        root_to_save_model=ROOT_TO_SAVE_MODEL,
        device=DEVICE,
        config=train_config
    )
    trainer.train_model()

    model_checkpoint = torch.load(os.path.join(ROOT_TO_SAVE_MODEL, 'checkpoint.pt'), map_location=DEVICE)

    model = object_from_dict(model_checkpoint['config']['model'], feature_extracting=False)
    model.freeze_only_first_n_layers(num_first_layers=train_config.num_freeze_layers_on_2nd_stage)
    model.load_state_dict(model_checkpoint['model'])
    model.to(DEVICE)

    optimizer = object_from_dict(train_config.optimizer,
                                 params=filter(lambda x: x.requires_grad, model.get_parameters()))

    criterion = object_from_dict(train_config.criterion)

    scheduler = object_from_dict(train_config.scheduler2, optimizer=optimizer)

    print('2nd stage')

    trainer = object_from_dict(
        train_config.trainer2,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        validloader=valid_loader,
        testloader=test_loader,
        root_to_save_model=ROOT_TO_SAVE_MODEL,
        device=DEVICE,
        config=train_config
    )
    # trainer.train_model()

    model_checkpoint = torch.load(os.path.join(ROOT_TO_SAVE_MODEL, 'checkpoint.pt'), map_location=DEVICE)

    model = object_from_dict(model_checkpoint['config']['model'], feature_extracting=False)
    model.freeze_only_first_n_layers(num_first_layers=train_config.num_freeze_layers_on_3rd_stage)
    model.load_state_dict(model_checkpoint['model'])
    model.to(DEVICE)

    optimizer = object_from_dict(train_config.optimizer,
                                 params=filter(lambda x: x.requires_grad, model.get_parameters()))

    criterion = object_from_dict(train_config.criterion)

    scheduler = object_from_dict(train_config.scheduler3, optimizer=optimizer)

    print('3rd stage')

    trainer = object_from_dict(
        train_config.trainer3,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        validloader=valid_loader,
        testloader=test_loader,
        root_to_save_model=ROOT_TO_SAVE_MODEL,
        device=DEVICE,
        config=train_config
    )
    # trainer.train_model()

    model_checkpoint = torch.load(os.path.join(ROOT_TO_SAVE_MODEL, 'checkpoint.pt'), map_location=DEVICE)

    model = object_from_dict(model_checkpoint['config']['model'], feature_extracting=False)

    model.load_state_dict(model_checkpoint['model'])
    model.to(DEVICE)

    optimizer = object_from_dict(train_config.optimizer,
                                 params=filter(lambda x: x.requires_grad, model.get_parameters()))

    criterion = object_from_dict(train_config.criterion)

    scheduler = object_from_dict(train_config.scheduler4, optimizer=optimizer)

    print('4th stage')

    trainer = object_from_dict(
        train_config.trainer4,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=train_loader,
        validloader=valid_loader,
        testloader=test_loader,
        root_to_save_model=ROOT_TO_SAVE_MODEL,
        device=DEVICE,
        config=train_config
    )
    trainer.train_model()


if __name__ == "__main__":
    train_cfg = get_param_from_config(args.config)
    main(train_cfg)
