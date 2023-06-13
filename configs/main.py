"""
Example of how to trigger an experiment
python main.py --config=configs/main.py:quadratic-sim_gda-sgd-sgd
python main.py --config=configs/main.py:bilinear-alt_gda-omd-omd --config.train.x.optimizer=OMEGA --config.train.y.optimizer=OMEGA

Choices of game: quadratic, quadratic_linear, bilinear
Choices of train method: sim_gda, alt_gda, extra_gda
Choices of optimizers: sgd, omd, extra_gda
"""

from configs.basic import build_basic_config
from configs.game import bilinear_config, quadratic_config, quadratic_linear_config
from configs.optim import (
    alt_gda_config,
    extra_gda_config,
    omd_config,
    omega_config,
    omegam_config,
    sgd_config,
    sim_gda_config,
)


def get_config(config_string):
    game_name, train_method, x_optimizer, y_optimizer = config_string.split("-")

    config = build_basic_config()
    config.game = globals()[game_name + "_config"]()
    config.train = globals()[train_method + "_config"]()
    config.train.x = globals()[x_optimizer + "_config"]()
    config.train.y = globals()[y_optimizer + "_config"]()

    return config
