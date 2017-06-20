# coding=utf-8

import os


def mkdirs_if_not_exist(path):
    """
    if path not exists, use os.makedirs(path) to mkdir recursively.
    """
    if not os.path.exists(path):
        os.makedirs(path)