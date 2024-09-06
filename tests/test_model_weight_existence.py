import os
from pytorchcv.models.common.model_store import get_model_metainfo_dict, get_model_file


def test_model_weight_existence(local_model_store_dir_path: str = os.path.join("~", ".torch", "models"),
                                remove: bool = True):
    model_metainfo_dict = get_model_metainfo_dict()
    for model_name, model_metainfo in model_metainfo_dict.items():
        print("model: {} -- ".format(model_name), end="")
        if model_metainfo[1] == "NA":
            print("skipped")
            continue
        file_path = get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path)
        if remove:
            os.remove(file_path)
        print("passed")
