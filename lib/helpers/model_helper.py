'''model_helper'''
from lib.models.occulde3d import build_occlude3d


def build_model(cfg):
    return build_occlude3d(cfg)
