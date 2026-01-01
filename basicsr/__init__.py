# flake8: noqa
from .version import __version__

# 导入子模块以支持动态注册
from basicsr import data
from basicsr import models
from basicsr import metrics
from basicsr import utils
