import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def sgd_config():
    _config = mlc.ConfigDict()
    _config.optimizer = "SGD"
    _config.lr = 1e-4
    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.momentum = 0.0
    _config.kwargs.nesterov = False
    _config.kwargs.dampening = 0.0

    return _config


def omd_config():
    _config = mlc.ConfigDict()
    _config.optimizer = "OMD"
    _config.lr = 1e-4
    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.optimism = 1.0

    return _config


def omega_config():
    _config = mlc.ConfigDict()
    _config.optimizer = "OMEGA"
    _config.lr = 1e-4
    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.optimism = 1.0
    _config.kwargs.ema_beta = 0.9

    return _config


def omegam_config():
    _config = mlc.ConfigDict()
    _config.optimizer = "OMEGAM"
    _config.lr = 1e-4
    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.optimism = 1.0
    _config.kwargs.ema_beta = 0.9

    return _config


def sim_gda_config():
    _config = mlc.ConfigDict()

    _config.method = "Simultaneous"
    _config.num_iters = 1000
    _config.batch_size = 1

    _config.x = mlc.ConfigDict()
    _config.y = mlc.ConfigDict()

    _config.use_lr_scheduler = False

    return _config


def alt_gda_config():
    _config = mlc.ConfigDict()

    _config.method = "Alternating"
    _config.num_iters = 1000
    _config.batch_size = 1

    _config.x = mlc.ConfigDict()
    _config.y = mlc.ConfigDict()

    _config.use_lr_scheduler = False

    return _config


def extra_gda_config():
    _config = mlc.ConfigDict()

    _config.method = "Extrapolation"
    _config.num_iters = 1000
    _config.batch_size = 1

    _config.x = mlc.ConfigDict()
    _config.y = mlc.ConfigDict()

    _config.use_lr_scheduler = False

    return _config
