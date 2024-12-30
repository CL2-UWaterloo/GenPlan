import json
from typing import Union

import numpy as np

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON.
    Reference: https://github.com/openai/spinningup
    """
    try:
        # the object is json serializable, just return it
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v)
                for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}
        return str(obj)


class RunningAverage(object):
    """Computes running mean and standard deviation.
    Reference: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    """

    def __init__(self, mean=0., vars=0., count=0) -> None:
        self.mean, self.vars = mean, vars
        self.count = count

    def reset(self):
        self.count = 0

    def add(self, x: Union[int, float]) -> None:
        """Add a number to the running average, update mean/std/count."""
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.vars = 0.
        else:
            prev_mean = self.mean
            self.mean += (x - self.mean) / self.count
            self.vars += (x - prev_mean) * (x - self.mean)

    def __add__(self, other):
        assert isinstance(other, RunningAverage)
        sum_ns = self.count + other.count
        prod_ns = self.count * other.count
        delta2 = (other.mean - self.mean)**2.
        return RunningAverage(
            (self.mean * self.count + other.mean * other.count) / sum_ns,
            self.vars + other.vars + delta2 * prod_ns / sum_ns, sum_ns
        )

    @property
    def var(self):
        return self.vars / (self.count) if self.count else 0.0

    @property
    def std(self):
        return np.sqrt(self.var)

    def __repr__(self):
        # return '<RunningAverage(mean={: 2.4f}, std={: 2.4f}, count={: 2f})>'.format(
        #     self.mean, self.std, self.count)
        return '{: .3g}'.format(self.mean)

    def __str__(self):
        return 'mean={: .3g}, std={: .3g}'.format(self.mean, self.std)

    def __call__(self):
        return self.mean


def test():
    from collections import defaultdict
    running_averages = [defaultdict(RunningAverage) for _ in range(2)]
    data = np.arange(10)
    for d in data[:5]:
        running_averages[0]["k"].add(d)
    print(running_averages[0]["k"])
    print(
        "numpy mean={: 2.4f}, std={: 2.4f}".format(np.mean(data[:5]), np.std(data[:5]))
    )

    for d in data[5:]:
        running_averages[1]["k"].add(d)
    print(running_averages[1]["k"])
    print(
        "numpy mean={: 2.4f}, std={: 2.4f}".format(np.mean(data[5:]), np.std(data[5:]))
    )

    print("Testing summation")
    print(running_averages[0]["k"] + running_averages[1]["k"])
    print("numpy mean={: 2.4f}, std={: 2.4f}".format(np.mean(data), np.std(data)))


import atexit
import json
import os
import os.path as osp
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
import yaml

class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer.  All the loggers
    create four panels by default: `train`, `test`, `loss`, and `update`.  Try to
    overwrite write() method to customize your own logger.

    :param str log_dir: the log directory. Default to None.
    :param bool log_txt: whether to log data in ``log_dir`` with name ``progress.txt``.
        Default to True.
    :param str name: the experiment name. If None, it will use the current time as the
        name. Default to None.
    """

    def __init__(self, log_dir=None, log_txt=True, name=None) -> None:
        super().__init__()
        self.name = name if name is not None else time.strftime("%Y-%m-%d_exp")
        self.log_dir = osp.join(log_dir, name) if log_dir is not None else None
        self.log_fname = "progress.txt"
        if log_dir:
            if osp.exists(self.log_dir):
                warning_msg = colorize(
                    "Warning: Log dir %s already exists! Some logs may be overwritten." %
                    self.log_dir, "magenta", True
                )
                print(warning_msg)
            else:
                os.makedirs(self.log_dir)
            if log_txt:
                self.output_file = open(osp.join(self.log_dir, self.log_fname), 'w')
                atexit.register(self.output_file.close)
                print(
                    colorize(
                        "Logging data to %s" % self.output_file.name, 'green', True
                    )
                )
        else:
            self.output_file = None
        self.first_row = True
        self.checkpoint_fn = None
        self.reset_data()

    def setup_checkpoint_fn(self, checkpoint_fn: Optional[Callable] = None) -> None:
        """Setup the function to obtain the model checkpoint, it will be called \
            when using ```logger.save_checkpoint()```.

        :param Optional[Callable] checkpoint_fn: the hook function to get the \
            checkpoint dictionary, defaults to None.
        """
        self.checkpoint_fn = checkpoint_fn

    def reset_data(self) -> None:
        """Reset stored data"""
        self.log_data = defaultdict(RunningAverage)

    def store(self, tab: str = None, **kwargs) -> None:
        """Store any values to the current epoch buffer with prefix `tab/`.

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs) logger.save_config(locals())

        :param str tab: the prefix of the logging data, defaults to None.
        """
        for k, v in kwargs.items():
            if tab is not None:
                k = tab + "/" + k
            self.log_data[k].add(np.mean(v))

    def write(
        self,
        step: int,
        display: bool = False,
        display_keys: Iterable[str] = None
    ) -> None:
        """Writing data to somewhere and reset the stored data.

        :param int step: the current training step or epochs
        :param bool display: whether print the logged data in terminal, default to False
        :param Iterable[str] display_keys: a list of keys to be printed. If None, print
            all stored keys, default to None.
        """
        if "update/env_step" not in self.logger_keys:
            self.store(tab="update", env_step=step)
        # save .txt file to the output logger
        if self.output_file is not None:
            if self.first_row:
                keys = ["Steps"] + list(self.logger_keys)
                self.output_file.write("\t".join(keys) + "\n")
            vals = [step] + self.get_mean_list(self.logger_keys)
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
            self.first_row = False
        if display:
            self.display_tabular(display_keys=display_keys)
        self.reset_data()

    def write_without_reset(self, *args, **kwarg) -> None:
        """Writing data to somewhere without resetting the current stored stats, \
            for tensorboard and wandb logger usage."""

    def save_checkpoint(self, suffix: Optional[Union[int, str]] = None) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param Optional[Union[int, str]] suffix: the suffix to be added to the stored
            checkpoint name, defaults to None.
        """
        if self.checkpoint_fn and self.log_dir:
            fpath = osp.join(self.log_dir, "checkpoint")
            os.makedirs(fpath, exist_ok=True)
            suffix = '%d' % suffix if isinstance(suffix, int) else suffix
            suffix = '_' + suffix if suffix is not None else ""
            fname = 'model' + suffix + '.pt'
            torch.save(self.checkpoint_fn(), osp.join(fpath, fname))

    def save_config(self, config: dict, verbose=True) -> None:
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important config
        vars as a dict. This will serialize the config to JSON, while handling anything
        which can't be serialized in a graceful way (writing as informative a string as
        possible).

        Example use:

        .. code-block:: python

            logger = BaseLogger(**logger_kwargs) logger.save_config(locals())

        :param dict config: the configs to be stored.
        :param bool verbose: whether to print the saved configs, default to True.
        """
        if self.name is not None:
            config['name'] = self.name
        config_json = convert_json(config)
        if verbose:
            print(colorize('Saving config:\n', color='cyan', bold=True))
            output = json.dumps(
                config_json, separators=(',', ':\t'), indent=4, sort_keys=True
            )
            print(output)
        if self.log_dir:
            with open(osp.join(self.log_dir, "config.yaml"), 'w') as out:
                yaml.dump(
                    config, out, default_flow_style=False, indent=4, sort_keys=False
                )

    def restore_data(self) -> None:
        """Return the metadata from existing log. Not implemented for BaseLogger.
        """
        pass

    def get_std(self, key: str) -> float:
        """Get the standard deviation of the queried data in storage.

        :param str key: the key of the queried data.
        :return: the standard deviation.
        """
        return self.log_data[key].std

    def get_mean(self, key: str) -> float:
        """Get the mean of the queried data in storage.

        :param str key: the key of the queried data.
        :return: the mean.
        """
        return self.log_data[key].mean

    def get_mean_list(self, keys: Iterable[str]) -> list:
        """Get the list of queried data in storage.

        :param Iterable[str] keys: the keys of the queried data.
        :return: the list of mean values.
        """
        return [self.get_mean(key) for key in keys]

    def get_mean_dict(self, keys: Iterable[str]) -> dict:
        """Get the dict of queried data in storage.

        :param Iterable[str] keys: the keys of the queried data.

        :return: the dict of mean values.
        """
        return {key: self.get_mean(key) for key in keys}

    @property
    def stats_mean(self) -> dict:
        return self.get_mean_dict(self.logger_keys)

    @property
    def logger_keys(self) -> Iterable:
        return self.log_data.keys()

    def display_tabular(self, display_keys: Iterable[str] = None) -> None:
        """Display the keys of interest in a tabular format.

        :param Iterable[str] display_keys: the keys to be displayed, if None, display
            all data. defaults to None.
        """
        if not display_keys:
            display_keys = sorted(self.logger_keys)
        key_lens = [len(key) for key in self.logger_keys]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in display_keys:
            val = self.log_data[key].mean
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
        print("-" * n_slashes, flush=True)

    def print(self, msg: str, color='green') -> None:
        """Print a colorized message to stdout.

        :param str msg: the string message to be printed
        :param str color: the colors for printing, the choices are ```gray, red, green,
            yellow, blue, magenta, cyan, white, crimson```. Default to "green".
        """
        print(colorize(msg, color, bold=True))


class DummyLogger(BaseLogger):
    """A logger that inherent from the BaseLogger but does nothing. \
         Used as the placeholder in trainer."""

    def __init__(self, *args, **kwarg) -> None:
        pass

    def setup_checkpoint_fn(self, *args, **kwarg) -> None:
        """The DummyLogger saves nothing"""

    def store(self, *args, **kwarg) -> None:
        """The DummyLogger stores nothing"""

    def reset_data(self, *args, **kwarg) -> None:
        """The DummyLogger resets nothing"""

    def write(self, *args, **kwarg) -> None:
        """The DummyLogger writes nothing."""

    def write_without_reset(self, *args, **kwarg) -> None:
        """The DummyLogger writes nothing"""

    def save_checkpoint(self, *args, **kwarg) -> None:
        """The DummyLogger saves nothing"""

    def save_config(self, *args, **kwarg) -> None:
        """The DummyLogger saves nothing"""

    def restore_data(self, *args, **kwarg) -> None:
        """The DummyLogger restores nothing"""

    def get_mean(self, *args, **kwarg) -> float:
        """The DummyLogger returns 0"""
        return 0

    def get_std(self, *args, **kwarg) -> float:
        """The DummyLogger returns 0"""
        return 0

    def get_mean_list(self, *args, **kwarg) -> None:
        """The DummyLogger returns nothing"""

    def get_mean_dict(self, *args, **kwarg) -> None:
        """The DummyLogger returns nothing"""

    @property
    def stats_mean(self) -> None:
        """The DummyLogger returns nothing"""

    @property
    def logger_keys(self) -> None:
        """The DummyLogger returns nothing"""
        
        

import uuid
from typing import Iterable

import wandb

class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    A typical usage example: ::

        config = {...} project = "test_cvpo" group = "SafetyCarCircle-v0" name =
        "default_param" log_dir = "logs"

        logger = WandbLogger(config, project, group, name, log_dir)
        logger.save_config(config)

        agent = CVPOAgent(env, logger=logger) agent.learn(train_envs)

    :param str config: experiment configurations. Default to an empty dict.
    :param str project: W&B project name. Default to "fsrl".
    :param str group: W&B group name. Default to "test".
    :param str name: W&B experiment run name. If None, it will use the current time as
        the name. Default to None.
    :param str log_dir: the log directory. Default to None.
    :param bool log_txt: whether to log data in ``log_dir`` with name ``progress.txt``.
        Default to True.
    """

    def __init__(
        self,
        config: dict = {},
        project: str = "fsrl",
        group: str = "test",
        name: str = None,
        log_dir: str = "log",
        log_txt: bool = True
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.wandb_run = wandb.init(
            project=project,
            group=group,
            name=name,
            id=str(uuid.uuid4()),
            resume="allow",
            config=config,  # type: ignore
        ) if not wandb.run else wandb.run
        # wandb.run.save()

    def write(
        self,
        step: int,
        display: bool = True,
        display_keys: Iterable[str] = None
    ) -> None:
        """Writing data to somewhere and reset the stored data.

        :param int step: the current training step or epochs
        :param bool display: whether print the logged data in terminal, default to False
        :param Iterable[str] display_keys: a list of keys to be printed. If None, print
            all stored keys, default to None.
        """
        self.store(tab="update", env_step=step)
        self.write_without_reset(step)
        return super().write(step, display, display_keys)

    def write_without_reset(self, step: int) -> None:
        """Sending data to wandb without resetting the current stored stats."""
        wandb.log(self.stats_mean, step=step)
    
    def log(self, data, step=None, commit=True):
        wandb.log(data, step=step, commit=commit)
    
    def finish(self):
        wandb.finish()

    def restore_data(self) -> None:
        """Not implemented yet"""