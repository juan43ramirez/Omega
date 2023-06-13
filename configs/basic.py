import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def make_game_config():
    _config = mlc.ConfigDict()

    _config.problem_name = MLC_PH(str)
    _config.dim = MLC_PH(int)
    _config.num_samples = MLC_PH(int)
    _config.bias = MLC_PH(bool)
    _config.make_normal_B = MLC_PH(bool)

    # Conditioning of the problem
    _config.mu_B = MLC_PH(float)
    _config.L_B = MLC_PH(float)
    _config.mu = MLC_PH(float)
    _config.L = MLC_PH(float)

    # Reproducibility
    _config.data_seed = MLC_PH(int)
    _config.sample_seed = MLC_PH(int)

    return _config


def make_train_config():
    _config = mlc.ConfigDict()
    _config.method = MLC_PH(str)
    _config.num_iters = MLC_PH(int)
    _config.tol = MLC_PH(float)
    _config.batch_size = MLC_PH(int)
    _config.use_lr_scheduler = MLC_PH(bool)

    _config.x = make_player_optim_config()
    _config.y = make_player_optim_config()

    return _config


def make_player_optim_config():
    _config = mlc.ConfigDict()

    _config.optimizer = MLC_PH(str)
    _config.lr = MLC_PH(float)
    _config.kwargs = mlc.ConfigDict()

    return _config


def build_basic_config():
    config = mlc.ConfigDict()

    # Populate top-level configs which are common to all experiments
    config.game = make_game_config()
    config.train = make_train_config()

    config.tag = "basic"

    # Fixed defaults for logging across all experiments
    config.logging = mlc.ConfigDict()
    config.logging.log_level = "INFO"
    config.logging.wandb_mode = "online"
    config.logging.log_freq = 100

    return config
