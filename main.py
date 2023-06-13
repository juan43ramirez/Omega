import logging

import torch
from absl import app
from absl.flags import FLAGS
from ml_collections.config_flags import config_flags as MLC_FLAGS

import wandb
from src import games, optim, utils

# Initialize "FLAGS.config" object with a placeholder config
MLC_FLAGS.DEFINE_config_file("config", default="configs/basic.py")

logging.basicConfig()
logger = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.utils.backcompat.broadcast_warning.enabled = True


def main(_):
    config = FLAGS.config
    logger.setLevel(getattr(logging, config.logging.log_level))
    wandb.init(project="omega", config=config.to_dict(), mode=config.logging.wandb_mode)

    dim = config.game.dim
    utils.seed_all(0)
    x = torch.nn.Parameter(torch.randn(dim, device=DEVICE))
    y = torch.nn.Parameter(torch.randn(dim, device=DEVICE))

    # config.game.name must be one of "Quadratic", "QuadraticLinear", "Bilinear"
    game_class = games.__dict__["Stochastic" + config.game.problem_name + "Game"]
    kwargs = config.game.to_dict()
    kwargs.pop("problem_name")
    game = game_class(**kwargs)
    game.to_(DEVICE)

    logger.info(f"Game: {game}")

    # Log initial values
    loss = game.loss(game.sample(config.train.batch_size), x, y)
    suboptimality = torch.abs(loss - game.f_star).detach()
    dist = game.dist2opt(x, y).detach()
    wandb.log({"suboptimality": suboptimality, "dist2opt": dist, "x": x.detach(), "y": y.detach()}, step=0)

    # Create the optimizers and wrap them in an OptimizerWrapper object
    lr, kwargs = config.train.x.lr, config.train.x.kwargs
    x_optimizer = optim.__dict__[config.train.x.optimizer]([x], lr=lr, **kwargs)

    lr, kwargs = config.train.y.lr, config.train.y.kwargs
    y_optimizer = optim.__dict__[config.train.y.optimizer]([y], lr=lr, **kwargs)

    joint_optimizer = optim.__dict__[config.train.method + "Optimizer"](x_optimizer, y_optimizer)

    mu = config.game.mu if "mu" in config.game else 0
    mu = max(mu, config.game.mu_B)

    decay_fn = (lambda step: 2 / (mu * (step + 1))) if config.train.use_lr_scheduler else (lambda step: 1)
    x_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(joint_optimizer.x_optimizer, decay_fn)
    y_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(joint_optimizer.y_optimizer, decay_fn)

    for i in range(config.train.num_iters):
        joint_optimizer.zero_grad()
        sample = game.sample(config.train.batch_size)
        loss = game.loss(sample, x, y)
        loss.backward()

        if config.train.method == "Simultaneous":
            joint_optimizer.step()
        else:
            # For alternating and extra-gradient methods, `joint_optimizer.step()` will
            # handle the re-calculation of gradients after the x update/extrapolation.
            joint_optimizer.step(closure=game.loss, idx=sample, x=x, y=y)

        x_lr_scheduler.step()
        y_lr_scheduler.step()

        dist = game.dist2opt(x, y).detach()
        suboptimality = torch.abs(loss - game.f_star).detach()

        if i % config.logging.log_freq == 0:
            logger.info(f"Epoch {i} / {config.train.num_iters}")
            logger.info(f"Suboptimality: {suboptimality}")
            logger.info(f"Distance to optimum: {dist}")

        current_lr = x_lr_scheduler.get_last_lr()[0]
        to_log = {"suboptimality": suboptimality, "dist2opt": dist, "x": x.detach(), "y": y.detach(), "lr": current_lr}
        wandb.log(to_log, step=i + 1)

        if torch.abs(dist) > 1e6:
            logger.info("Diverged")
            break
        if torch.abs(dist) < 1e-20:
            logger.info("Converged with tolerance {}".format(1e-20))
            break

    state_dict = game.state_dict()
    torch.save(state_dict, "game.pt")
    wandb.save("game.pt")


if __name__ == "__main__":
    app.run(main)
