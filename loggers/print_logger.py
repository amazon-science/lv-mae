import os

from .base_logger import BaseLogger
from typing import Dict, Any, List, Optional
from pprint import pprint


class PrintLogger(BaseLogger):

    def __init__(self, *args, **kwargs):
        super(PrintLogger, self).__init__(*args, **kwargs)
        from PIL.Image import Image
        from matplotlib import pyplot as plt
        import numpy as np
        import torch as th
        self.Image = Image
        self.plt = plt
        self.np = np
        self.th = th

    def stop(self):
        pass

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            print(f'{name}: {data}' if step is None else f'step {step}, {name}: {data:.4e}')

    def _log_fig(self, name: str, fig: Any):
        if self.rank == 0:
            if isinstance(fig, self.Image):
                fig = self.np.asarray(fig)
                self.plt.imshow(fig)
            elif isinstance(fig, self.np.ndarray):
                # higher resolution
                sizes = self.np.shape(fig)
                plt_fig = self.plt.figure()
                plt_fig.set_size_inches((1. * sizes[1] / sizes[0])*10, 10, forward=False)
                ax = self.plt.Axes(plt_fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                plt_fig.add_axes(ax)
                ax.imshow(fig)
                self.plt.show()
            elif isinstance(fig, self.th.Tensor):
                fig = fig.cpu().detach().numpy()
                sizes = self.np.shape(fig)
                plt_fig = self.plt.figure()
                plt_fig.set_size_inches((1. * sizes[1] / sizes[0])*10, 10, forward=False)
                ax = self.plt.Axes(plt_fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                plt_fig.add_axes(ax)
                ax.imshow(fig)
                self.plt.show()
            else:
                self.plt.show()

    def log_hparams(self, params: Dict[str, Any]):
        if self.rank == 0:
            print('hyperparameters:')
            pprint(params)

    def log_params(self, params: Dict[str, Any]):
        if self.rank == 0:
            print('params:')
            pprint(params)

    def add_tags(self, tags: List[str]):
        if self.rank == 0:
            print('tags:')
            pprint(tags)

    def log_name_params(self, name: str, params: Any):
        if self.rank == 0:
            print(f'{name}:')
            pprint(params)


class TqdmLogger(PrintLogger):

    def __init__(self, *args, **kwargs):
        super(TqdmLogger, self).__init__(*args, **kwargs)
        from tqdm import tqdm
        self.tqdm = tqdm

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            self.tqdm.write(f'{name}: {data}' if step is None else f'step {step}, {name}: {data:.4e}')


class LoggerL(PrintLogger):

    def __init__(self, stdout, format=None, *args, **kwargs):
        super(LoggerL, self).__init__(*args, **kwargs)
        from matplotlib import pyplot as plt
        import logging
        self.show = plt.show
        handler = logging.StreamHandler(stdout)
        if format is None:
            format = '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'
        handler.setFormatter(logging.Formatter(format))
        self.logger = logging.getLogger()
        self.logger.addHandler(handler)
        self.logger.setLevel('INFO')
        self.logging = logging

    def log(self, text: str, data: Any, step=None):
        if self.rank == 0:
            self.logging.info(text % data)