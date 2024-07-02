from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.model_store import get_model_stats, calc_net_weights


def test_model_weight_counts(pretrained: bool = False):
    model_stats_dict = get_model_stats()
    for model_name, model_stats in model_stats_dict.items():
        net = ptcv_get_model(model_name, pretrained=pretrained)
        net.eval()
        net_weight_count = calc_net_weights(net)
        model_weight_count = model_stats[0]
        if model_weight_count == 0:
            break
        assert (net_weight_count == model_weight_count)
