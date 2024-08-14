from __future__ import annotations

import os
from os import path as osp
from shutil import copyfile

import torch

from .logger import Logger


def save_checkpoint(state: object, filename: str, logger: Logger) -> str:
    if osp.isfile(filename):
        if hasattr(logger, "log"):
            logger.log(f"Find {filename} exist, delete is at first before saving")
        os.remove(filename)
    torch.save(state, filename)
    assert osp.isfile(
        filename
    ), f"save filename : {filename} failed, which is not found."
    if hasattr(logger, "log"):
        logger.log(f"save checkpoint into {filename}")
    return filename


def copy_checkpoint(src: str, dst: str, logger: Logger) -> None:
    if osp.isfile(dst):
        if hasattr(logger, "log"):
            logger.log(f"Find {dst} exist, delete is at first before saving")
        os.remove(dst)
    copyfile(src, dst)
    if hasattr(logger, "log"):
        logger.log(f"copy the file from {src} into {dst}")
