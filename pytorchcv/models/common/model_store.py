"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_metainfo_dict', 'get_model_file', 'load_model', 'download_model', 'calc_net_weight_count',
           'get_model_weight_count']

import os
import zipfile
import logging
import hashlib
import torch.nn as nn

imgclsmob_repo_url = "https://github.com/osmr/imgclsmob"
model_metainfos_file_name = "model_metainfos.csv"


def get_model_metainfos_file_path() -> str:
    """
    Get file path for the `model_metainfos` file.

    Returns
    -------
    str
        File path.
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), model_metainfos_file_name)


def load_csv(csv_file_path: str) -> list[list[str]]:
    """
    The simplest loader from CSV.

    Parameters
    ----------
    csv_file_path : str
        Source file path.

    Returns
    -------
    list(list(str))
        CSV data.
    """
    from csv import reader
    with open(csv_file_path, "r") as file:
        data = list(reader(file))
    return data


def conv_str_to_int(value: str) -> int:
    """
    Convert string ot int.

    Parameters
    ----------
    value : str
        Imput value.

    Returns
    -------
    int
        Output value.
    """
    return int(value) if value != "NA" else 0


def get_model_metainfo_dict() -> dict[str, tuple[int, str, str, str]]:
    """
    Get model metainfos from CSV file.

    Returns
    -------
    dict(str, tuple(str, int, str, str))
        CSV data.
    """
    model_metainfos_file_path = get_model_metainfos_file_path()
    model_metainfos_list = load_csv(model_metainfos_file_path)
    assert all([len(x) == 12 for x in model_metainfos_list])
    assert all([x[2] == "NA" or len(x[2]) == 4 for x in model_metainfos_list[1:]])
    model_metainfo_dict = {x[0]: (conv_str_to_int(x[1]), x[2], x[3], x[4]) for x in model_metainfos_list[1:]}
    return model_metainfo_dict


def get_model_metainfo(model_name: str) -> tuple[int, str, str, str]:
    """
    Get model metainfo for particular model.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    tuple(int, str, str, str)
        Trainable weight count.
    """
    model_metainfo_dict = get_model_metainfo_dict()
    if model_name not in model_metainfo_dict:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    return model_metainfo_dict[model_name]


def get_model_weight_count(model_name: str) -> int:
    """
    Get model trainable weight count.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    int
        Trainable weight count.
    """
    net_weights, error_value, sha1_hash, repo_release_tag = get_model_metainfo(model_name)
    return net_weights


def get_model_name_suffix_data(model_name: str) -> tuple[str, str, str]:
    """
    Get model name suffix strings.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    tuple(str, str, str)
        Model name suffix strings.
    """
    net_weights, error_value, sha1_hash, repo_release_tag = get_model_metainfo(model_name)
    return error_value, sha1_hash, repo_release_tag


def get_model_file(model_name: str,
                   local_model_store_dir_path: str = os.path.join("~", ".torch", "models")) -> str:
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    str
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.pth".format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info("Model file not found. Downloading to {}.".format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(
        url="{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip".format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")


def _download(url: str,
              path: str | None = None,
              overwrite: bool = False,
              sha1_hash: str | None = None,
              retries: int = 5,
              verify_ssl: bool = True) -> str:
    """
    Download a given URL.

    Parameters
    ----------
    url : str
        URL to download
    path : str or None, default None
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, default False
        Whether to overwrite destination file if already exists.
    sha1_hash : str or None, default None
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. " \
            "Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


def _check_sha1(file_name: str,
                sha1_hash: str) -> bool:
    """
    Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(file_name, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def load_model(net: nn.Module,
               file_path: str,
               ignore_extra: bool = True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    net : nn.Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path, weights_only=False)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


def download_model(net: nn.Module,
                   model_name: str,
                   local_model_store_dir_path: str = os.path.join("~", ".torch", "models"),
                   ignore_extra: bool = True):
    """
    Load model state dictionary from a file with downloading it if necessary.

    Parameters
    ----------
    net : nn.Module
        Network in which weights are loaded.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    load_model(
        net=net,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path),
        ignore_extra=ignore_extra)


def calc_net_weight_count(net: nn.Module) -> int:
    """
    Calculate network trainable weight/parameters count.

    Parameters
    ----------
    net : nn.Module
        Network.

    Returns
    -------
    int
        Calculated number of weights/parameters.
    """
    # import numpy as np
    from functools import reduce
    from operator import mul
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        # weight_count += np.prod(param.size())
        weight_count += reduce(mul, param.size())
    return weight_count
