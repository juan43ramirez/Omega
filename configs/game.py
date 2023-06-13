import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def bilinear_config():
    _config = mlc.ConfigDict()

    _config.problem_name = "Bilinear"
    _config.dim = 1
    _config.num_samples = 1000
    _config.bias = True
    _config.make_normal_B = False

    # Conditioning of the problem
    _config.mu_B = 1.0
    _config.L_B = 1.0

    # Reproducibility
    _config.data_seed = 0
    _config.sample_seed = 0

    return _config


def quadratic_linear_config():
    _config = mlc.ConfigDict()

    _config.problem_name = "QuadraticLinear"
    _config.dim = 1
    _config.num_samples = 1000
    _config.bias = True
    _config.make_normal_B = False

    # Conditioning of the problem
    _config.mu_B = 1.0
    _config.L_B = 1.0
    _config.mu = 1.0
    _config.L = 1.0

    # Reproducibility
    _config.data_seed = 0
    _config.sample_seed = 0

    return _config


def quadratic_config():
    _config = mlc.ConfigDict()

    _config.problem_name = "Quadratic"
    _config.dim = 1
    _config.num_samples = 1000
    _config.bias = True
    _config.make_normal_B = False

    # Conditioning of the problem
    _config.mu_B = 1.0
    _config.L_B = 1.0
    _config.mu = 1.0
    _config.L = 1.0

    # Reproducibility
    _config.data_seed = 0
    _config.sample_seed = 0

    return _config
