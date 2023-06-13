import torch


def make_sym_matrix(n_samples, dim, mu=0.0, L=1.0, num_zeros=0):
    """Make n_samples of symmetric semi-positive definite matrices, all with eigenvalues
    in [mu, L]. Used for the individual terms A, C of the game

    num_zeros: to reduce the rank of the matrix, set some eigenvalues to zero.
    """

    A = torch.rand(n_samples, dim, dim)
    Q, _ = torch.linalg.qr(A)
    eigs = torch.zeros(n_samples, dim)
    random_eig = torch.rand(n_samples, dim - num_zeros)
    if random_eig.numel() > 1:
        max_eig = random_eig.max()  # 1, keepdim=True)[0]
        min_eig = random_eig.min()  # 1, keepdim=True)[0]
        random_eig = (random_eig - min_eig) / (max_eig - min_eig)
    else:
        random_eig[:] = 1
    eigs[:, : dim - num_zeros] = mu + random_eig * (L - mu)
    X = torch.bmm(torch.bmm(Q, torch.diag_embed(eigs)), Q.transpose(-1, -2))
    return X


def make_random_matrix(n_samples, dim, mu=0.0, L=1.0, normal=False):
    """Make n_samples of random matrices, each with eigenvalues in [mu, L].

    Matrices are not necessarily symmetric, so they can be used for the coupling term
    of the game.

    normal: if True, the matrix is normal, i.e. U^T U = UU^T = I
    """

    A = torch.rand(n_samples, dim, dim)
    U, S, V = torch.svd(A)
    S = torch.rand(n_samples, dim)
    if S.numel() > 1:
        S_min = S.min()  # 1, keepdim=True)[0]
        S_max = S.max()  # 1, keepdim=True)[0]
        S = (S - S_min) / (S_max - S_min)
    else:
        S[:] = 1
    S = mu + S * (L - mu)

    if normal:
        V = U.transpose(-2, -1)

    X = torch.bmm(torch.bmm(U, torch.diag_embed(S)), V)
    return X
