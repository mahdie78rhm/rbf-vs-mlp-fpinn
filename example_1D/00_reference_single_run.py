"""Reference implementation (single run, seed 1234) of the 1D time-fractional diffusion benchmark.

RBF-fPINN and MLP-fPINN are trained once with the settings of the reference paper
(5000 Adam + up to 20000 L-BFGS iterations). Used as the starting point of Stage 0."""

import torch
import torch.nn as nn
import math
import numpy as np
from scipy.special import gamma

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# Exact solution and RHS
def exact_u(X, mu):
    t, x = X[:, [0]], X[:, [1]]

    return (1 + t**3) * torch.exp(-mu * (x - 0.5) ** 2)


def source_term(X, alpha, mu):
    t, x = X[:, [0]], X[:, [1]]
    phi = torch.exp(-mu * (x - 0.5) ** 2)

    ut = 3 * t**2 * phi
    Dtu = phi * (6.0 / math.gamma(4 - alpha)) * t ** (3 - alpha)
    uxx = (1 + t**3) * (-2 * mu + 4 * mu**2 * (x - 0.5) ** 2) * phi

    return ut + Dtu - uxx


# Creating data
def data_train():
    t = torch.from_numpy(np.linspace(lb[0], ub[0], num_t)[:, None]).float()
    x_data = torch.from_numpy(np.linspace(lb[1], ub[1], num_x)[:, None]).float()

    return t, x_data


def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_test_N)[:, None]
    t_star, x_star = np.meshgrid(t_test, x_test)
    test_data = np.hstack((t_star.flatten()[:, None], x_star.flatten()[:, None]))
    test_data = torch.from_numpy(test_data).float().to(device)
    test_exact = exact_u(test_data, mu)

    return t_test, x_test, test_data, test_exact


# RBF-fPINN Network
class BasisFunction(nn.Module):
    """A single class to handle different basis functions dynamically."""

    def __init__(self, func_type="gaussian"):
        super(BasisFunction, self).__init__()
        self.func_type = func_type.lower()

    def forward(self, alpha):
        """Select and apply the appropriate basis function."""
        if self.func_type == "gaussian":
            return torch.exp(-alpha.pow(2))
        elif self.func_type == "linear":
            return alpha
        elif self.func_type == "quadratic":
            return alpha.pow(2)
        elif self.func_type == "inverse quadratic":
            return torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        elif self.func_type == "multiquadric":
            return torch.sqrt(torch.ones_like(alpha) + alpha.pow(2))
        elif self.func_type == "inverse multiquadric":
            return torch.ones_like(alpha) / torch.sqrt(torch.ones_like(alpha) + alpha.pow(2))
        elif self.func_type == "spline":
            return alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha))
        elif self.func_type == "poisson one":
            return (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
        elif self.func_type == "poisson two":
            return (
                ((alpha - 2 * torch.ones_like(alpha)) / (2 * torch.ones_like(alpha)))
                * alpha
                * torch.exp(-alpha)
            )
        elif self.func_type == "matern32":
            sqrt3 = math.sqrt(3)
            return (torch.ones_like(alpha) + sqrt3 * alpha) * torch.exp(-sqrt3 * alpha)
        elif self.func_type == "matern52":
            sqrt5 = math.sqrt(5)
            return (torch.ones_like(alpha) + sqrt5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(
                -sqrt5 * alpha
            )
        else:
            raise ValueError(f"Unknown basis function type: {self.func_type}")


class RBF(nn.Module):
    def __init__(self, in_features, out_features, basis_func_type="gaussian"):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = BasisFunction(basis_func_type)  # Unified basis function class
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centres, 0.0, 1.0)
        nn.init.constant_(self.log_sigmas, math.log(0.2))

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        sq_distances = (x - c).pow(2).sum(-1)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0)

        # For Gaussian, avoid sqrt() because u_xx is obtained through autograd.
        if self.basis_func.func_type == "gaussian":
            return torch.exp(-sq_distances / sigma.pow(2))

        # Other radial bases use r = ||x-c|| / sigma.
        distances = torch.sqrt(sq_distances + torch.finfo(input.dtype).eps) / sigma
        return self.basis_func(distances)


class Net(nn.Module):
    def __init__(self, in_features, hidden_neurons, out_features, basis_func_type="gaussian"):
        super(Net, self).__init__()
        self.iter = 0
        self.rbf = RBF(in_features, hidden_neurons, basis_func_type)
        self.linear = nn.Linear(hidden_neurons, out_features, bias=False)

    def forward(self, x):
        out = self.rbf(x)
        out = self.linear(out)
        return out


# fPINN Network
class Net_fpinn(nn.Module):
    def __init__(self, layers):
        super(Net_fpinn, self).__init__()
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        a = x
        for i in range(len(self.linear) - 1):
            a = self.activation(self.linear[i](a))
        return self.linear[-1](a)


class Model:
    def __init__(self, net, x_data, t, lb, ub, test_data, test_exact, alpha):

        self.net = net
        self.alpha = alpha

        self.x_data = x_data.to(device)
        self.t = t.to(device)
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.test_data = test_data.to(device)
        self.test_exact = test_exact.to(device)

        self.num_x = len(x_data)
        self.num_t = len(t)
        self.dt = (ub[0] - lb[0]) / (self.num_t - 1)

        self.coef = self.coef_a(np.arange(self.num_t), self.alpha)
        self.init_data()

    def coef_a(self, n, alpha):
        return self.dt ** (-alpha) * ((n + 1) ** (1 - alpha) - n ** (1 - alpha)) / gamma(2 - alpha)

    def init_data(self):
        temp_t0 = torch.full((self.num_x, 1), self.t[0][0], device=device)
        self.tx_t0 = torch.cat((temp_t0, self.x_data), dim=1)

        temp_t = self.t.clone().detach().to(dtype=torch.float32, device=device)
        x_data_repeated = self.x_data.repeat((self.num_t, 1))
        self.tx = torch.cat(
            (temp_t.repeat_interleave(self.num_x).view(-1, 1), x_data_repeated), dim=1
        )

        temp_lb = torch.full((self.num_t, 1), self.lb[1], device=device)
        temp_ub = torch.full((self.num_t, 1), self.ub[1], device=device)
        self.tx_b1 = torch.cat((temp_t, temp_lb), dim=1)
        self.tx_b2 = torch.cat((temp_t, temp_ub), dim=1)

        self.u_x_b1 = exact_u(self.tx_b1, mu)
        self.u_x_b2 = exact_u(self.tx_b2, mu)
        self.u_t0 = exact_u(self.tx_t0, mu)
        self.F_tx = source_term(self.tx, self.alpha, mu)

    def train_U(self, x):
        # The physical domain is already [0, 1]^2; no input scaling is needed.
        return self.net(x)

    def PDE_loss(self):
        coef = self.coef
        x = self.tx.clone().detach().requires_grad_(True)
        u_pred = self.train_U(x).to(device)
        u_t = torch.autograd.grad(
            u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True
        )[0][:, [0]]
        u_x = torch.autograd.grad(
            u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True
        )[0][:, [1]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][
            :, [1]
        ]

        u_n = u_pred.reshape(self.num_t, -1)
        Lu = -u_t.reshape(self.num_t, -1) + u_xx.reshape(self.num_t, -1)
        F_n = self.F_tx.reshape(self.num_t, -1)

        loss = torch.tensor(0.0).to(device)

        for n in range(1, self.num_t):
            if n == 1:
                pre_Ui = (Lu[n] + F_n[n]) / coef[0] + (coef[n - 1] / coef[0]) * u_n[0]
            else:
                pre_Ui = ((Lu[n] + F_n[n]) / coef[0] + (coef[n - 1] / coef[0]) * u_n[0]).to(device)
                for k in range(1, n):
                    pre_Ui += ((coef[n - k - 1] - coef[n - k]) / coef[0]) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss / (self.num_t - 1)

    def compute_loss(self):
        loss_initial = torch.mean((self.train_U(self.tx_t0) - self.u_t0) ** 2)
        loss_boundary1 = torch.mean((self.train_U(self.tx_b1) - self.u_x_b1) ** 2)
        loss_boundary2 = torch.mean((self.train_U(self.tx_b2) - self.u_x_b2) ** 2)
        loss_boundary = loss_boundary1 + loss_boundary2
        loss_pde = self.PDE_loss()

        return loss_initial, loss_boundary, loss_pde

    # LBFGS closure
    def LBFGS_loss(self):
        self.optimizer_LBFGS.zero_grad()
        loss_initial, loss_boundary, loss_pde = self.compute_loss()
        total_loss = loss_initial + loss_boundary + loss_pde
        total_loss.backward()
        self.net.iter += 1
        if self.net.iter % 100 == 0:
            print("Iter:", self.net.iter, "Loss:", total_loss.item())
        return total_loss

    def train(self, Adam_epochs=5000, LBFGS_epochs=50000):
        # ---------------- Adam warm-up phase ----------------
        # A short Adam phase before L-BFGS helps move the network into a
        # reasonable basin before the more locally-greedy L-BFGS takes over --
        # without it, L-BFGS (starting from a random init) is more prone to
        # getting stuck in a poor local minimum, especially for the non-convex,
        # multi-term (IC + BC + fractional-PDE) loss here.
        self.optimizer_Adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for _ in range(Adam_epochs):
            self.optimizer_Adam.zero_grad()
            loss_initial, loss_boundary, loss_pde = self.compute_loss()
            total_loss = loss_initial + loss_boundary + loss_pde
            total_loss.backward()
            self.optimizer_Adam.step()
            self.net.iter += 1
            if self.net.iter % 100 == 0:
                print("Iter:", self.net.iter, "Loss:", total_loss.item(), "(Adam)")

        # ---------------- L-BFGS phase ----------------
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=LBFGS_epochs,
            max_eval=LBFGS_epochs,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        self.optimizer_LBFGS.step(self.LBFGS_loss)
        print("-----------------Done------------------")

        with torch.no_grad():
            pred = self.train_U(self.tx).cpu().numpy()
            exact = exact_u(self.tx, mu).cpu().numpy()
            train_error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)

            test_pred = self.train_U(self.test_data).cpu().numpy()
            test_exact = self.test_exact.cpu().numpy()
            test_error = np.linalg.norm(test_pred - test_exact, 2) / np.linalg.norm(test_exact, 2)

        print("Train_L2error:", f"{train_error:.4e}")
        print("Test_L2error :", f"{test_error:.4e}")

        loss_initial, loss_boundary, loss_pde = self.compute_loss()
        final_loss = (loss_initial + loss_boundary + loss_pde).item()
        return train_error, test_error, final_loss


if __name__ == "__main__":
    set_seed(1234)
    alpha = 0.1

    # Width of the Gaussian profie
    mu = 320

    # Initialization
    n_inputs = 2
    n_outputs = 1
    n_layers = 1
    n_neurons = 150
    use_RBFfPINN = True

    # Choosing Network
    if use_RBFfPINN:
        net = Net(
            in_features=n_inputs,
            hidden_neurons=n_neurons,
            out_features=n_outputs,
            basis_func_type="gaussian",
        ).to(device)
        print("RBFfPINN network created.")
    else:
        layers = [n_inputs] + [n_neurons] * n_layers + [n_outputs]
        net = Net_fpinn(layers).to(device)
        print("FPINN network created.")

    lb = np.array([0.0, 0.0])  # low boundary
    ub = np.array([1.0, 1.0])  # up boundary

    """train data"""
    num_t = 50
    num_x = 50
    t, x_data = data_train()

    """test data"""
    t_test_N = 100
    x_test_N = 100
    t_test, x_test, test_data, test_exact = data_test()

    """Train"""
    model = Model(
        net=net,
        x_data=x_data,
        t=t,
        lb=lb,
        ub=ub,
        test_data=test_data,
        test_exact=test_exact,
        alpha=alpha,
    )

    model.train(Adam_epochs=5000, LBFGS_epochs=20000)
