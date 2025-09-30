import warnings
import os
from pathlib import Path

from .base_logger import BaseLogger
from typing import Dict, Any, List
import numpy as np
from PIL import Image


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}


def convert_no_basic_to_str_from_any(p: Any):
    if is_basic(p):
        return p
    elif isinstance(p, dict):
        return convert_no_basic_to_str(p)
    else:
        return str(p)


class WandbLogger(BaseLogger):

    def __init__(self, project=None, *args, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        if self.rank != 0:
            return
        import wandb
        self.wandb = wandb
        from pathlib import Path
        # home_path_api_token = Path.home() / '.wandb' / 'token.txt'
        SRC = os.path.join(os.path.realpath(__file__).split('v2_lvm')[0], 'v2_lvm')
        home_path_api_token = Path(os.path.join(SRC, 'wandb/token.txt'))
        local_path_api_token = Path('wandb') / 'token.txt'
        if local_path_api_token.exists():
            api_token = local_path_api_token
        elif home_path_api_token.exists():
            api_token = home_path_api_token
        else:
            warnings.warn('''Please create a file at .wandb/token.txt with your Wandb API token.
            Or add a file at wandb/token.txt''')
            raise FileNotFoundError('Wandb token not found')

        api_token = api_token.read_text().strip()
        if project is None:
            home_path_api_project = Path(os.path.join(SRC, 'wandb/project.txt'))
            local_path_api_project = Path('wandb') / 'project.txt'
            if local_path_api_project.exists():
                project = local_path_api_project.read_text().strip()
            elif home_path_api_project.exists():
                project = home_path_api_project.read_text().strip()
            else:
                warnings.warn('''Please create a file at wandb/project.txt with your Wandb project name''')
                raise FileNotFoundError('Wandb project not found')
        wandb.login(key=api_token)
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project=project,
        )

    def stop(self):
        if self.rank == 0:
            self.run.finish()

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            self.wandb.log({name: data})

    def _log_fig(self, name: str, fig: Any):
        if self.rank == 0:
            if isinstance(fig, np.ndarray):
                if fig.dtype != np.uint8:
                    fig = fig * 255
                    fig = fig.astype(np.uint8)
                fig = Image.fromarray(fig)
            self.wandb.log({name: self.wandb.Image(fig)})

    def log_hparams(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            if isinstance(params, dict):
                self.wandb.config.update(params)
            else:
                self.wandb.config.update({'hparams': params})


    def log_params(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            if isinstance(params, dict):
                self.wandb.config.update(params)
            else:
                self.wandb.config.update({'params': params})

    def add_tags(self, tags: List[str]):
        if self.rank == 0:
            self.run.tags = self.run.tags + tuple(tags)

    def log_name_params(self, name: str, params: Any):
        if self.rank == 0:
            params = convert_no_basic_to_str_from_any(params)
            self.wandb.config.update({name: params}, allow_val_change=True)