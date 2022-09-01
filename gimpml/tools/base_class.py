import os, tempfile, traceback, json, torch
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from importlib.util import find_spec


class ModelBase(ABC):
    def __init__(self, config: dict):
        """ Inits predictor configuration. """
        self.hub_repo = None
        self.device = torch.device(
            "cuda" if not ("force_cpu" in config and config["force_cpu"]) and (
                torch.cuda.is_available()) else "cpu")
        self.n_images = config["n_drawables"] if "n_drawables" in config else False
        self.mask_name = config["mask_name"] if "mask_name" in config else None
        self.model_name = config["model_name"] if "model_name" in config else None
        self._model = None
        self._rpc = None

    @property
    def model(self):
        """ Loads model from local file or hub configuration for predictor. """
        if self._model is None:
            if find_spec("tqdm"):
                with capture_tqdm(self.update_progress, "Downloading model"):
                    self._model = self.load_model()
            else:
                self._model = self.load_model()
        return self._model

    @abstractmethod
    def load_model(self):
        """ Sets model configuration for predictor. """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        """ Runs the current model on a given images. """
        raise NotImplementedError

    def update_progress(self, percent, message):
        if self._rpc:
            self._rpc.update_progress(percent, message)


class capture_tqdm:
    """ Custom tqdm progress bar class. """
    
    def __init__(self, progress_fn, default_desc=None):
        self.progress_fn = progress_fn
        self.default_desc = default_desc

    def __enter__(self):
        from tqdm import tqdm
        self.tqdm_display = tqdm.display

        def custom_tqdm_display(tqdm_self, *args, **kwargs):
            tqdm_info = tqdm_self.format_dict.copy()
            tqdm_info["prefix"] = tqdm_info["prefix"] or self.default_desc
            tqdm_info["bar_format"] = "{desc}: " if tqdm_info["prefix"] else ""
            # Removed {percentage:3.0f}% from bar_format
            tqdm_info["bar_format"] += "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            message = tqdm_self.format_meter(**tqdm_info)
            percent = None
            if tqdm_info["total"]:
                percent = tqdm_info["n"] / float(tqdm_info["total"])
            self.progress_fn(percent, message)
            self.tqdm_display(tqdm_self, *args, **kwargs)

        tqdm.display = custom_tqdm_display
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        from tqdm import tqdm
        tqdm.display = self.tqdm_display

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator