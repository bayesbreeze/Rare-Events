# Import libraries
import numpy as np
# import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# %matplotlib inline
import time
import pylab as pl
from IPython import display
from tqdm.auto import trange, tqdm
# from google.colab import files

# Data hyper-parameters
N = 1024  # nr of datapoints

# Model hyper-parameters
init_gamma_0 = -13.3  # initial gamma_0
init_gamma_1 = 5.0  # initial gamma_1
hidden_units = 512
T_train = 0  # nr of timesteps in model; T=0 means continuous-time
vocab_size = 256

# Optimization hyper-parameters
learning_rate = 3e-3
num_train_steps = 20000  # nr of training steps

rng = jax.random.PRNGKey(seed=0)

# Make 8-bit swirl dataset
np.random.seed(0)
theta = np.sqrt(np.random.rand(N)) * 3 * np.pi  # np.linspace(0,2*pi,100)
r_a = 2 * theta + np.pi
x = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
# We use 8 bits, to make this a bit similar to image data, which has 8-bit
# color channels.
x = 4 * (x + 0.25 * np.random.randn(N, 2) + 30)
x = x.astype("uint8")
# plt.scatter(x[:, 0], x[:, 1], alpha=0.1)
# plt.show()

# Get mean and standard deviation of 'x'
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)

# Define learnable model
class Model(nn.Module):
    def setup(self):
        self.score_net = ScoreNetwork()
        self.noise_schedule = NoiseSchedule()

    def __call__(self, x, t):
        gamma_t = self.noise_schedule(t)
        return self.score_net(x, gamma_t)

    def score(self, x, t):
        return self.score_net(x, t)

    def gamma(self, t):
        return self.noise_schedule(t)


class ScoreNetwork(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(hidden_units)
        self.dense2 = nn.Dense(hidden_units)
        self.dense3 = nn.Dense(2)
        self.ff = Base2FourierFeatures()

    def __call__(self, z, gamma_t):

        # Normalize gamma_t
        lb = init_gamma_0
        ub = init_gamma_1
        gamma_t_norm = ((gamma_t - lb) / (ub - lb)) * 2 - 1  # ---> [-1,+1]

        # Concatenate normalized gamma_t as extra feature
        h = jnp.concatenate([z, gamma_t_norm[:, None]], axis=1)

        # append Fourier features
        h_ff = self.ff(h)
        h = jnp.concatenate([h, h_ff], axis=1)

        # Three dense layers
        h = nn.swish(self.dense1(h))
        h = nn.swish(self.dense2(h))
        h = self.dense3(h)

        return h


class Base2FourierFeatures(nn.Module):
    # Create Base 2 Fourier features
    @nn.compact
    def __call__(self, inputs):
        freqs = jnp.asarray(range(8), dtype=inputs.dtype)  # [0, 1, ..., 7]
        w = 2.0**freqs * 2 * jnp.pi
        w = jnp.tile(w[None, :], (1, inputs.shape[-1]))
        h = jnp.repeat(inputs, len(freqs), axis=-1)
        h *= w
        h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
        return h


def constant_init(value, dtype="float32"):
    def _init(key, shape, dtype=dtype):
        return value * jnp.ones(shape, dtype)

    return _init


# Simple scalar noise schedule, i.e. gamma(t) in the paper:
# gamma(t) = abs(w) * t + b
class NoiseSchedule(nn.Module):
    def setup(self):
        init_bias = init_gamma_0
        init_scale = init_gamma_1 - init_gamma_0
        self.w = self.param("w", constant_init(init_scale), (1,))
        self.b = self.param("b", constant_init(init_bias), (1,))

    def __call__(self, t):
        return abs(self.w) * t + self.b


def data_encode(x):
    # This transforms x from discrete values (0, 1, ...)
    # to the domain (-1,1).
    # Rounding here just a safeguard to ensure the input is discrete
    # (although typically, x is a discrete variable such as uint8)
    x = x.round()
    return (x - x_mean) / x_std


def data_decode(z_0_rescaled, gamma_0):
    # Logits are exact if there are no dependencies between dimensions of x
    x_vals = jnp.arange(0, vocab_size)[:, None]
    x_vals = jnp.repeat(x_vals, z_0_rescaled.shape[-1], 1)
    x_vals = data_encode(x_vals).transpose([1, 0])[None, :, :]
    inv_stdev = jnp.exp(-0.5 * gamma_0[..., None])
    logits = -0.5 * jnp.square((z_0_rescaled[..., None] - x_vals) * inv_stdev)

    logprobs = jax.nn.log_softmax(logits)
    return logprobs


def data_logprob(x, z_0_rescaled, gamma_0):
    x = x.round().astype("int32")
    x_onehot = jax.nn.one_hot(x, vocab_size)
    logprobs = data_decode(z_0_rescaled, gamma_0)
    logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2))
    return logprob


def data_generate_x(z_0, gamma_0, rng):
    var_0 = nn.sigmoid(gamma_0)
    z_0_rescaled = z_0 / jnp.sqrt(1.0 - var_0)
    logits = data_decode(z_0_rescaled, gamma_0)
    samples = jax.random.categorical(rng, logits)
    return samples


# define loss function
def loss_fn(params, x, rng):

    gamma = lambda t: model.apply(params, t, method=Model.gamma)
    gamma_0, gamma_1 = gamma(0.0), gamma(1.0)
    var_0, var_1 = nn.sigmoid(gamma_0), nn.sigmoid(gamma_1)
    n_batch = x.shape[0]

    # encode
    f = data_encode(x)

    # 1. RECONSTRUCTION LOSS
    # add noise and reconstruct
    rng, rng1 = jax.random.split(rng)
    eps_0 = jax.random.normal(rng1, shape=f.shape)
    z_0 = jnp.sqrt(1.0 - var_0) * f + jnp.sqrt(var_0) * eps_0
    z_0_rescaled = f + jnp.exp(0.5 * gamma_0) * eps_0  # = z_0/sqrt(1-var)
    loss_recon = -data_logprob(x, z_0_rescaled, gamma_0)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = (1.0 - var_1) * jnp.square(f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1.0, axis=1)

    # 3. DIFFUSION LOSS
    # sample time steps
    rng, rng1 = jax.random.split(rng)
    t = jax.random.uniform(rng1, shape=(n_batch,))

    # discretize time steps if we're working with discrete time
    if T_train > 0:
        t = jnp.ceil(t * T_train) / T_train

    # sample z_t
    gamma_t = gamma(t)
    var_t = nn.sigmoid(gamma_t)[:, None]
    rng, rng1 = jax.random.split(rng)
    eps = jax.random.normal(rng1, shape=f.shape)
    z_t = jnp.sqrt(1.0 - var_t) * f + jnp.sqrt(var_t) * eps
    # compute predicted noise
    eps_hat = model.apply(params, z_t, gamma_t, method=Model.score)
    # compute MSE of predicted noise
    loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=1)

    if T_train == 0:
        # loss for infinite depth T, i.e. continuous time
        _, g_t_grad = jax.jvp(gamma, (t,), (jnp.ones_like(t),))
        loss_diff = 0.5 * g_t_grad * loss_diff_mse
    else:
        # loss for finite depth T, i.e. discrete time
        s = t - (1.0 / T_train)
        gamma_s = gamma(s)
        loss_diff = 0.5 * T_train * jnp.expm1(gamma_t - gamma_s) * loss_diff_mse

    # End of diffusion loss computation

    # Compute loss in terms of bits per dimension
    rescale_to_bpd = 1.0 / (np.prod(x.shape[1:]) * np.log(2.0))
    bpd_latent = jnp.mean(loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    loss = bpd
    metrics = [bpd_latent, bpd_recon, bpd_diff]
    return loss, metrics


# define training step
@jax.jit
def train_step(rng, optim_state, params, x):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    rng, rng1 = jax.random.split(rng)
    (loss, metrics), grads = grad_fn(params, x, rng1)
    updates, optim_state = optimizer.update(grads, optim_state, params)
    params = optax.apply_updates(params, updates)
    return rng, optim_state, params, loss, metrics


# Initialize model
model = Model()
rng, rng1, rng2 = jax.random.split(rng, 3)
init_inputs = [128 * jnp.ones((1, 2)), jnp.zeros((1,))]
params = model.init({"params": rng1, "sample": rng2}, *init_inputs)

# initialize optimizer
optimizer = optax.adamw(learning_rate)
optim_state = optimizer.init(params)

# training loop (takes 12 mins on a TPU)
losses = []
for i in trange(num_train_steps):
    rng, optim_state, params, loss, _metrics = train_step(rng, optim_state, params, x)
    losses.append(loss)

print("~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~")