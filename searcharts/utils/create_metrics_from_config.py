from searcharts.utils import object_from_dict


def get_metrics_from_config(config_metrics: list) -> list:
    all_metrics = []
    for config_metric in config_metrics:
        dict_metric_type = config_metric[0]
        for param in config_metric[1].params:
            metric = object_from_dict(dict_metric_type, **param[0])
            all_metrics.append(metric)
    return all_metrics