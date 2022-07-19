import os
import torch
import argparse
import pickle

from searcharts.models import ArtEfficientnet, Embedder, SimilaritySearch
from searcharts.data import lock_deterministic, get_train_aug_preproc, get_valid_aug_preproc
from searcharts.utils import get_param_from_config, object_from_dict
from searcharts.models import Evaluator
from searcharts.utils import get_metrics_from_config


SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
config_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../configs"))

parser = argparse.ArgumentParser()
parser.add_argument('--config', action="store", dest="config",
                    default=os.path.join(config_dir, "train_classifier.yaml"),
                    type=str)

args = parser.parse_args()


def main(train_config: dict):
    VALID_PATH = train_config.valid_path
    TRAIN_PATH = train_config.train_path
    ROOT = train_config.root
    updates_per_epoch = train_config.updates_per_epoch

    DEVICE = torch.device(train_config.device)
    PREPROCESSING_FN = ArtEfficientnet.get_preprocess_fn()

    ROOT_TO_SAVE_MODEL = train_config.root_to_save_model

    lock_deterministic(train_config.seed)


    train_dataset = object_from_dict(train_config.dataset, csv_path=TRAIN_PATH, root_to_data=ROOT,
                                     augmentation=get_train_aug_preproc(PREPROCESSING_FN), mode='train',
                                     length=updates_per_epoch * train_config.train_dataloader.batch_size)
    train_loader = train_dataset.get_dataloader(**train_config.train_dataloader, shuffle=True)

    valid_dataset = object_from_dict(train_config.dataset, csv_path=VALID_PATH, root_to_data=ROOT,
                                     augmentation=get_valid_aug_preproc(PREPROCESSING_FN), mode='val')
    valid_loader = valid_dataset.get_dataloader(**train_config.valid_dataloader, shuffle=False)

    test_index_loader = object_from_dict(train_config.dataset, csv_path=train_config.test_index_loader_path,
                                         root_to_data=ROOT,
                                         augmentation=get_valid_aug_preproc(PREPROCESSING_FN), mode='val') \
        .get_dataloader(**train_config.test_index_loader, shuffle=False)

    test_search_loader = object_from_dict(train_config.dataset, csv_path=train_config.test_search_loader_path,
                                          root_to_data=ROOT,
                                          augmentation=get_valid_aug_preproc(PREPROCESSING_FN), mode='val') \
        .get_dataloader(**train_config.test_search_loader, shuffle=False)

    if not os.path.exists(os.path.join(train_config.root_to_save_model)):
        os.mkdir(os.path.join(train_config.root_to_save_model))

    with open(os.path.join(train_config.root_to_save_model, "labels.pickle"), 'wb') as f:
        pickle.dump(train_dataset.labels, f)

    embedding_size = train_config.embedding_size
    classes = train_config.classes

    model = object_from_dict(train_config.model, vector_size=embedding_size).to(DEVICE)

    metric_fc = object_from_dict(train_config.loss, out_features=classes, in_features=embedding_size, device=DEVICE).to(
        DEVICE)

    model_optimizer = object_from_dict(train_config.model_optimizer,
                                       params=filter(lambda x: x.requires_grad, model.get_cnn_parameters()))

    fc_optimizer = object_from_dict(train_config.fc_optimizer, params=[
        {'params': filter(lambda x: x.requires_grad, metric_fc.parameters())},
        {'params': filter(lambda x: x.requires_grad, model.get_fc_parameters())}
    ])

    criterion = object_from_dict(train_config.criterion)

    model_scheduler = object_from_dict(train_config.model_scheduler, optimizer=model_optimizer)

    fc_scheduler = object_from_dict(train_config.fc_scheduler, optimizer=fc_optimizer)

    embedder = Embedder(model)

    index = object_from_dict(train_config.index, dimension=embedding_size, device=DEVICE)

    similarity_search = SimilaritySearch(
        embedder=embedder,
        index=index,
        csv_file_with_images_paths=train_config.test_index_loader_path,
        label_columns=train_dataset.label_columns,
        labels=train_dataset.labels,
        device=DEVICE,
    )
    metrics = get_metrics_from_config(train_config.evaluator.metrics)

    evaluator = Evaluator(similarity_search=similarity_search, metrics=metrics, device=DEVICE)

    trainer = object_from_dict(
        train_config.trainer,
        test_index_loader=test_index_loader,
        test_search_loader=test_search_loader,
        model=model,
        metric_model=metric_fc,
        criterion=criterion,
        model_optimizer=model_optimizer, fc_optimizer=fc_optimizer,
        model_scheduler=model_scheduler, fc_scheduler=fc_scheduler,
        trainloader=train_loader,
        validloader=valid_loader,
        root_to_save_model=ROOT_TO_SAVE_MODEL,
        device=DEVICE,
        config=train_config,
        evaluator=evaluator
    )

    trainer.train_model()


if __name__ == "__main__":
    train_cfg = get_param_from_config(args.config)
    main(train_cfg)
