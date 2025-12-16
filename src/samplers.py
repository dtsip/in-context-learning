import enum
import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "ar1":AR1Sampler,
        "vr1":VAR1Sampler,
        "sparse_gaussian": SparseGaussianSampler,
        "ar2":AR2Sampler,
        "vr2":VR2Sampler,
        "nonstation":NonStationarySampler,
        "uniform": UniformSampler,
        "exponential": ExponentialSampler,
        "laplace": LaplaceSampler,
        "gamma": GammaSampler,
        "beta": BetaSampler,
        "tstudent": TStudentSampler,
        "poisson": PoissonSampler,
        "rayleigh": RayleighSampler,
        "cauchy": CauchySampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        # Only add 'k' parameter for sparse_gaussian sampler
        if data_name == "sparse_gaussian" and 'k' not in kwargs:
            kwargs['k'] = n_dims // 2  # default k is half of dimensions
        # Only add 'scale' parameter for sparse_gaussian sampler (as scalar)
        if data_name == "sparse_gaussian" and 'scale' not in kwargs:
            kwargs['scale'] = 1.0      # default scale is 1.0 for sparse_gaussian
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

def _sample_distribution(dist, b_size, inner_shape, seeds=None, device="cpu"):
    sample_shape = (b_size, *inner_shape)
    if seeds is None:
        samples = dist.sample(sample_shape)
        return samples.to(device) if device != "cpu" else samples

    assert len(seeds) == b_size
    template = dist.mean
    xs_b = torch.empty(sample_shape, dtype=template.dtype, device=device)
    for i, seed in enumerate(seeds):
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            xs_b[i] = dist.sample(inner_shape).to(device)
    return xs_b

class UniformSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, low=0.0, high=1.0):
        super().__init__(n_dims)
        self.bias = bias 
        self.scale = scale
        self.low = low
        self.high = high
    
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        uni_dist = torch.distributions.Uniform(self.low, self.high)
        xs_b = _sample_distribution(uni_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class ExponentialSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, rate=1.0):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.rate = float(rate)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        exp_dist = torch.distributions.Exponential(rate=self.rate)
        xs_b = _sample_distribution(exp_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class LaplaceSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, loc=0.0, laplace_scale=1.0):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.loc = float(loc)
        self.laplace_scale = float(laplace_scale)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        laplace_dist = torch.distributions.Laplace(loc=self.loc, scale=self.laplace_scale)
        xs_b = _sample_distribution(laplace_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class GammaSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, concentration=2.0, rate=1.0):
        super().__init__(n_dims)
        if concentration <= 0 or rate <= 0:
            raise ValueError("concentration and rate must be positive for Gamma distribution.")
        self.bias = bias
        self.scale = scale
        self.concentration = float(concentration)
        self.rate = float(rate)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        gamma_dist = torch.distributions.Gamma(concentration=self.concentration, rate=self.rate)
        xs_b = _sample_distribution(gamma_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims, device=device)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)
            generator = torch.Generator(device=device)
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator, device=device)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class BetaSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, alpha=2.0, beta=5.0):
        super().__init__(n_dims)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive for Beta distribution.")
        self.bias = bias
        self.scale = scale
        self.alpha = float(alpha)
        self.beta = float(beta)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        beta_dist = torch.distributions.Beta(concentration1=self.alpha, concentration0=self.beta)
        xs_b = _sample_distribution(beta_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
    
class TStudentSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, df=3.0):
        super().__init__(n_dims)
        self.df = float(df)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        t_dist = torch.distributions.StudentT(df=self.df)
        xs_b = _sample_distribution(t_dist, b_size, (n_points, self.n_dims), seeds, device)
        
        if self.scale is not None:
            xs_b = xs_b * self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class PoissonSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, rate=1.0):
        super().__init__(n_dims)
        self.rate = float(rate)
        self.bias = bias
        self.scale = scale
        
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        poisson_dist = torch.distributions.Poisson(rate=self.rate)
        xs_b = _sample_distribution(poisson_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class RayleighSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, scale_param=1.0):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.scale_param = float(scale_param)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        rayleigh_dist = torch.distributions.Rayleigh(scale=self.scale_param)
        xs_b = _sample_distribution(rayleigh_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class CauchySampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, loc=0.0, scale_param=1.0):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.loc = float(loc)
        self.scale_param = float(scale_param)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        cauchy_dist = torch.distributions.Cauchy(loc=self.loc, scale=self.scale_param)
        xs_b = _sample_distribution(cauchy_dist, b_size, (n_points, self.n_dims), seeds, device)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b

class SparseGaussianSampler(DataSampler):
    def __init__(self, n_dims, k, bias=None, scale=None):
        super().__init__(n_dims)
        if not (0 < k <= n_dims):
            raise ValueError(f"k must be in range (0, {n_dims}]")
        self.k = int(k)
        self.bias = bias
        # Store scale as float
        self.scale = float(scale) if isinstance(scale, (int, float)) else 1.0
    
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        if seeds is None:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)
            values = torch.randn(b_size, n_points, self.k, device=device)
            rand_scores = torch.rand(b_size, n_points, self.n_dims, device=device)
            _, indices = torch.topk(rand_scores, self.k, dim=-1)
            xs_b.scatter_(dim=2, index=indices, src=values)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)
            assert len(seeds) == b_size
            for i in range(b_size):
                generator = torch.Generator(device=device).manual_seed(int(seeds[i]))
                values = torch.randn(n_points, self.k, generator=generator, device=device)
                rand_scores = torch.rand(n_points, self.n_dims, generator=generator, device=device)
                _, indices = torch.topk(rand_scores, self.k, dim=-1)
                xs_b[i].scatter_(dim=1, index=indices, src=values)

        if self.scale is not None:
            # Simple scalar multiplication 
            xs_b = xs_b * self.scale
            
        if self.bias is not None:
            xs_b += self.bias
            
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
            
        return xs_b


class AR1Sampler(DataSampler):
    def __init__(self, n_dims, rho=0.9, noise_std=1.0, bias=None, scale=None, compute_gradient=False):
        super().__init__(n_dims)
        assert 0 <= abs(rho) < 1, "|rho| must be < 1 for a stable AR(1)"
        self.rho = float(rho)
        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale
        self.compute_gradient = compute_gradient

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        # Shape: (batch, time, dims)
        xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)

        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = []
            for seed in seeds:
                g = torch.Generator(device=device)
                g.manual_seed(int(seed))
                generators.append(g)

        # Initialize x_0 ~ N(0, I)
        if generators is None:
            xs_b[:, 0, :] = torch.randn(b_size, self.n_dims, device=device)
        else:
            for i in range(b_size):
                xs_b[i, 0, :] = torch.randn(self.n_dims, generator=generators[i], device=device)

        # AR(1): x_t = rho * x_{t-1} + eps_t, eps_t ~ N(0, noise_std^2 I)
        for t in range(1, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims, device=device)
            else:
                eps_t = torch.zeros(b_size, self.n_dims, device=device)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i], device=device)
            xs_b[:, t, :] = self.rho * xs_b[:, t - 1, :] + eps_t

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b

class AR2Sampler(DataSampler):
    def __init__(self, n_dims, ar1_coef=0.5, ar2_coef=0.3, noise_std=1.0, bias=None, scale=None):
        super().__init__(n_dims)
        assert abs(ar2_coef) < 1, "|ar2_coef| must be < 1 for a stable AR(2)"
        
        self.ar1_coef = float(ar1_coef)
        self.ar2_coef = float(ar2_coef)
        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        # Shape: (batch, time, dims)
        xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)

        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = []
            for seed in seeds:
                g = torch.Generator(device=device)
                g.manual_seed(int(seed))
                generators.append(g)

        # Initialize first two time steps
        for t in range(2):
            if generators is None:
                xs_b[:, t, :] = torch.randn(b_size, self.n_dims, device=device)
            else:
                for i in range(b_size):
                    xs_b[i, t, :] = torch.randn(self.n_dims, generator=generators[i], device=device)

        # AR(2): x_t = ar1_coef * x_{t-1} + ar2_coef * x_{t-2} + eps_t
        for t in range(2, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims, device=device)
            else:
                eps_t = torch.zeros(b_size, self.n_dims, device=device)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i], device=device)
            xs_b[:, t, :] = (
                self.ar1_coef * xs_b[:, t - 1, :] +
                self.ar2_coef * xs_b[:, t - 2, :] +
                eps_t
            )
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b

class VR2Sampler(DataSampler):
    def __init__(self, n_dims, ar1_mat=None, ar2_mat=None, noise_std=1.0, bias=None, scale=None):
        super().__init__(n_dims)
        
        if ar1_mat is None:
            ar1_mat = 0.5 * torch.eye(n_dims)
        if ar2_mat is None:
            ar2_mat = 0.3 * torch.eye(n_dims)
            
        # Check 
        assert ar1_mat.shape == (n_dims, n_dims), "ar1_mat must be n_dims x n_dims"
        assert ar2_mat.shape == (n_dims, n_dims), "ar2_mat must be n_dims x n_dims"
        
        self.ar1_mat = torch.tensor(ar1_mat, dtype=torch.float32)
        self.ar2_mat = torch.tensor(ar2_mat, dtype=torch.float32)
        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)

        generators = None
        if seeds is not None:
            generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

        # Initialize first two time points
        for t in range(2):
            if generators is None:
                xs_b[:, t, :] = torch.randn(b_size, self.n_dims, device=device)
            else:
                for i in range(b_size):
                    xs_b[i, t, :] = torch.randn(self.n_dims, generator=generators[i], device=device)
        
        # VR(2): x_t = A1 * x_{t-1} + A2 * x_{t-2} + eps_t
        for t in range(2, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims, device=device)
            else:
                eps_t = torch.zeros(b_size, self.n_dims, device=device)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i], device=device)
                    
            # Matrix multiplication for each sample in batch
            ar1_mat_device = self.ar1_mat.to(device)
            ar2_mat_device = self.ar2_mat.to(device)
            xs_b[:, t, :] = (torch.matmul(xs_b[:, t-1, :], ar1_mat_device.T) + 
                            torch.matmul(xs_b[:, t-2, :], ar2_mat_device.T) + 
                            eps_t)
            
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
            
        return xs_b

class NonStationarySampler(DataSampler):
    def __init__(self, n_dims, coef_base=0.5, coef_amplitude=0.4, noise_std=0.1, bias=None, scale=None):
        super().__init__(n_dims)
        self.coef_base = float(coef_base)
        self.coef_amplitude = float(coef_amplitude)
        self.noise_std = float(noise_std)
        self.scale = scale
        self.bias = bias
        
    def get_transition_matrix(self, t, n_points):
        t_norm = t / (n_points - 1) if n_points > 1 else 0.0
        time_varying_factor = self.coef_base + self.coef_amplitude * math.sin(2 * math.pi * t_norm)
        A_t = time_varying_factor * torch.eye(self.n_dims)
        return A_t

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)
        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
            
        if generators is None:
            xs_b[:,0,:] = torch.randn(b_size, self.n_dims, device=device) * self.noise_std
        else:
            for i in range(b_size):
                xs_b[i, 0, :] = torch.randn(self.n_dims, generator=generators[i], device=device) * self.noise_std
                
        for t in range(1, n_points):
            A_t = self.get_transition_matrix(t, n_points).to(device)

            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims, device=device)
            else:
                eps_t = torch.zeros(b_size, self.n_dims, device=device)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i], device=device)
            xs_b[:, t, :] = (torch.matmul(xs_b[:, t-1, :], A_t) + eps_t)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        
        return xs_b

class VAR1Sampler(DataSampler):
    def __init__(self, n_dims, ar1_mat=None, noise_std=1.0, bias=None, scale=None):
        super().__init__(n_dims)

        if ar1_mat is None:
            ar1_mat = 0.9 * torch.eye(n_dims)

        assert ar1_mat.shape == (n_dims, n_dims), "ar1_mat must be n_dims x n_dims"

        if isinstance(ar1_mat, torch.Tensor):
            self.ar1_mat = ar1_mat.float()
        else:
            self.ar1_mat = torch.tensor(ar1_mat, dtype=torch.float32)

        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale
        
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device="cpu"):
        xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)

        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

        if generators is None:
            xs_b[:, 0, :] = torch.randn(b_size, self.n_dims, device=device)
        else:
            for i in range(b_size):
                xs_b[i, 0, :] = torch.randn(self.n_dims, generator=generators[i], device=device)
                
        for t in range(1, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims, device=device)
            else:
                eps_t = torch.zeros(b_size, self.n_dims, device=device)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i], device=device)
            
            ar1_mat_device = self.ar1_mat.to(device)
            xs_b[:, t, :] = torch.matmul(xs_b[:, t - 1, :], ar1_mat_device.T) + eps_t

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
