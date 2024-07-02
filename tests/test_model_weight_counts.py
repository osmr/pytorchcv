from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.model_store import get_model_metainfo_dict, calc_net_weight_count


def test_model_weight_counts(pretrained: bool = False):
    model_metainfo_dict = get_model_metainfo_dict()
    for model_name, model_metainfo in model_metainfo_dict.items():
        net = ptcv_get_model(model_name, pretrained=pretrained)
        net.eval()
        net_weight_count = calc_net_weight_count(net)
        model_weight_count = model_metainfo[0]
        if model_weight_count == 0:
            break
        assert (net_weight_count == model_weight_count)
