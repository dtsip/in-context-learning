import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def huber_loss(ys_pred, ys, delta=1.35):
    """Huber loss - robust to outliers"""
    error = ys - ys_pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return (0.5 * quadratic.square() + delta * linear).mean()


def cauchy_loss(ys_pred, ys):
    """Cauchy loss - very robust to outliers (for Cauchy noise)"""
    error = ys - ys_pred
    return torch.log(1 + error.square()).mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "uniform_hypersphere_regression": UniformHypersphereRegression,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "ar1_linear_regression": AR1LinearRegression,
        "exponential_weighted_regression": ExponentialWeightedRegression,
        "laplace_weighted_regression": LaplaceWeightedRegression,
        "wlaplace_noisypoisson": wlaplace_noisypoisson,
        "sparse_regression_killer": SparseRegressionKiller,
        "heavy_tail_noise_killer": HeavyTailNoiseKiller,
        "bounded_support_killer": BoundedSupportKiller,
        "mixture_tasks_killer": MixtureTasksKiller,
        "transfer_tradeoff_task": TransferTradeoffTask,
    }

    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        
        # Simple return for all tasks - no special case needed
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class UniformHypersphereRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(UniformHypersphereRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            w_b = torch.randn(self.b_size, self.n_dims, 1)  
            self.w_b = w_b / w_b.norm(dim=1, keepdim=True)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                w = torch.randn(self.n_dims, 1, generator=generator)
                self.w_b[i] = w / torch.norm(w)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0] 
        # ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_linear

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        w = torch.randn(num_tasks, n_dims, 1)
        w_normalized = w / torch.norm(w, dim=1, keepdim=True)
        return {"w": w_normalized}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
class LaplaceWeightedRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, weight_scale=1.0):
        super(LaplaceWeightedRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.weight_scale = weight_scale # self.weight_scale as weight_scale

        if pool_dict is None and seeds is None:
            laplace_dist = torch.distributions.Laplace(loc=0, scale=self.weight_scale)
            self.w_b = laplace_dist.sample((self.b_size, self.n_dims, 1))
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                laplace_dist = torch.distributions.Laplace(loc=0, scale=self.weight_scale)
                self.w_b[i] = laplace_dist.sample((self.n_dims, 1))
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
            
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0] 
        ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, weight_scale=1.0):
        laplace_dist = torch.distributions.Laplace(loc=0, scale=weight_scale)
        return {"w": laplace_dist.sample((num_tasks, n_dims, 1))}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class wlaplace_noisypoisson(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1.0,
        weight_scale=1.0,
        poisson_rate=3.0,
    ):
        """
        Task with Laplace-distributed weights, expects exponential-like inputs,
        and adds centered Poisson noise to the supervision.
        """
        super(wlaplace_noisypoisson, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.weight_scale = weight_scale
        self.poisson_rate = float(poisson_rate)

        if pool_dict is None and seeds is None:
            laplace_dist = torch.distributions.Laplace(loc=0.0, scale=self.weight_scale)
            self.w_b = laplace_dist.sample((self.b_size, self.n_dims, 1))
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                laplace_dist = torch.distributions.Laplace(loc=0.0, scale=self.weight_scale)
                self.w_b[i] = laplace_dist.sample((self.n_dims, 1))
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0]

        poisson = torch.distributions.Poisson(rate=self.poisson_rate)
        noise = poisson.sample(ys_linear.shape) - self.poisson_rate
        noise = noise.to(xs_b.device)
        return ys_linear + noise

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, weight_scale=1.0):
        laplace_dist = torch.distributions.Laplace(loc=0.0, scale=weight_scale)
        return {"w": laplace_dist.sample((num_tasks, n_dims, 1))}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
class ExponentialWeightedRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rate=1.0):
        super(ExponentialWeightedRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.rate = rate

        if pool_dict is None and seeds is None:
            exp_dist = torch.distributions.Exponential(rate=self.rate)
            self.w_b = exp_dist.sample((self.b_size, self.n_dims, 1))
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                exp_dist = torch.distributions.Exponential(rate=self.rate)
                self.w_b[i] = exp_dist.sample((self.n_dims, 1))
        else: 
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0] 
        ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, rate=1.0):
        exp_dist = torch.distributions.Exponential(rate=rate)
        return {"w": exp_dist.sample((num_tasks, n_dims, 1))}
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return mean_squared_error
class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1,uniform=False):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            if uniform:
                self.w_b = torch.rand(self.b_size, self.n_dims, 1)*2 -1
            else:
                self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error



class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=2.0,
        renormalize_ys=False,
        noise_type="cauchy",  # "normal", "uniform", "laplace", "t-student", "cauchy", "exponential", "rayleigh", "beta", "poisson"        
        w_distribution="gaussian",
        w_kwargs=None,
        uniform=False,
    ):
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, uniform
        )
        self.noise_std = float(noise_std)
        self.renormalize_ys = renormalize_ys
        self.noise_type = noise_type.lower()
        self.w_distribution = w_distribution.lower()
        self.w_kwargs = w_kwargs or {}
        self.w_b = self._compose_weights(pool_dict, seeds)

    def _compose_weights(self, pool_dict, seeds):
        target_shape = (self.b_size, self.n_dims, 1)
        if pool_dict is not None:
            indices = torch.randperm(len(pool_dict["w"]))[: self.b_size]
            return pool_dict["w"][indices]
        
        if seeds is None:
            return self._sample_distribution(target_shape, generator=None)
        w_b = torch.zeros(target_shape)
        for i, seed in enumerate(seeds):
            gen = torch.Generator().manual_seed(int(seed))
            w_b[i] = self._sample_distribution((1, self.n_dims, 1), generator=gen).squeeze(0)
        return w_b
        
    def _sample_distribution(self, shape, generator=None, device='cpu'):
        def to_val(val):
            return torch.tensor(val, device=device) if not torch.is_tensor(val) else val.to(device)
        if self.w_distribution == "gaussian":
            scale = self.w_kwargs.get("scale", 1.0)
            return scale * torch.randn(shape, generator=generator, device=device)
        elif self.w_distribution == "uniform":
            low = self.w_kwargs.get("low", -1.0)
            high = self.w_kwargs.get("high", 1.0)
            return torch.empty(shape, generator=generator, device=device).uniform_(low, high)
        elif self.w_distribution == "laplace":
            scale = self.w_kwargs.get("scale", 1.0)
            laplace_dist = torch.distributions.Laplace(loc=0.0, scale=scale)
            return laplace_dist.sample(shape, generator=generator, device=device)
        elif self.w_distribution == "exponential":
            rate = self.w_kwargs.get("rate", 1.0)
            exp_dist = torch.distributions.Exponential(rate=rate)
            return exp_dist.sample(shape, generator=generator, device=device)
        elif self.w_distribution == "beta":
            alpha = self.w_kwargs.get("alpha", 2.0)
            beta = self.w_kwargs.get("beta", 5.0)
            beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
            return beta_dist.sample(shape, generator=generator, device=device)
        elif self.w_distribution == "poisson":
            rate = self.w_kwargs.get("rate", 3.0)
            dist = torch.distributions.Poisson(rate=rate)
            return dist.sample(shape, generator=generator, device=device)
        elif self.w_distribution == "cauchy":
            scale = self.w_kwargs.get("scale", 1.0)
            cauchy_dist = torch.distributions.StudentT(df=1, loc=0.0, scale=scale)
            return cauchy_dist.sample(shape, generator=generator, device=device)
        elif self.w_distribution == "t-student":
            df = self.w_kwargs.get("df", 3.0)
            scale = self.w_kwargs.get("scale", 1.0)
            t_dist = torch.distributions.StudentT(df=df, loc=0.0, scale=scale)
            return t_dist.sample(shape, generator=generator, device=device)
        elif self.w_distribution == "rayleigh":
            lambda_param = self.w_kwargs.get("lambda_param", 1.0)
            sigma = lambda_param
            X = torch.randn(shape, generator=generator, device=device) * sigma
            Y = torch.randn(shape, generator=generator, device=device) * sigma
            R = torch.sqrt(X**2 + Y**2)
            return R
        else: 
            raise ValueError(f"Unsupported weight distribution: {self.w_distribution}")
    def sample_noise(self, shape, device='cpu'):
        # 1.
        if self.noise_type == "normal":
            noise = torch.randn(shape, device=device) * self.noise_std
        # 2.
        elif self.noise_type == "uniform":
            a = math.sqrt(3) * self.noise_std
            noise = torch.empty(shape, device=device).uniform_(-a, a)
        # 3.
        elif self.noise_type == "laplace":
            scale_param = self.noise_std / math.sqrt(2.0)
            laplace_dist = torch.distributions.Laplace(loc=0, scale=scale_param)
            noise = laplace_dist.sample(shape, device=device)
        # 4.
        elif self.noise_type == "t-student":
            df = 3.0
            scale_param = self.noise_std / math.sqrt(df / (df-2.0))
            t_dist = torch.distributions.StudentT(df=df, loc=0, scale=scale_param)
            noise = t_dist.sample(shape, device=device)
        # 5.
        elif self.noise_type == "cauchy":
            scale_param = self.noise_std 
            cauchy_dist = torch.distributions.StudentT(df=1, loc=0, scale=scale_param)
            noise = cauchy_dist.sample(shape, device=device)   
        # 6.
        elif self.noise_type == "exponential":
            exp_noise = torch.distributions.Exponential(rate=1.0 / self.noise_std)
            noise = exp_noise.sample(shape, device=device) - self.noise_std
        # 7.
        elif self.noise_type == "rayleigh":
            lambda_param = self.noise_std / math.sqrt(2.0 - math.pi / 2.0)
            # R = sqrt(X^2 + Y^2) với X, Y ~ N(0, sigma^2), 
            # where sigma = lambda_param.
            sigma = lambda_param

            X = torch.randn(shape, device=device) * sigma
            Y = torch.randn(shape, device=device) * sigma
            R = torch.sqrt(X**2 + Y**2)
            mean = lambda_param * math.sqrt(math.pi / 2.0)
            noise = R - mean
        # 8.
        elif self.noise_type == "beta":
            alpha, beta = 2.0, 5.0
            mean = alpha / (alpha + beta)
            var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
            std = math.sqrt(var)
            beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
            X = beta_dist.sample(shape, device=device)
            noise = (X - mean) / std * self.noise_std
        # 9.
        elif self.noise_type == "poisson":
            lam = 3.0
            poisson_noise = torch.distributions.Poisson(lam)
            X = poisson_noise.sample(shape, device=device)
            scale_factor = self.noise_std / math.sqrt(lam)
            noise = (X - lam) * scale_factor     
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return noise

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        noise = self.sample_noise(ys_b.shape, device=ys_b.device)
        ys_b_noisy = ys_b + noise

        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy

    def get_training_metric(self):
        """
        Use robust loss for heavy-tailed noise (Cauchy, t-student) to handle outliers.
        For normal/uniform noise, use standard MSE.
        """
        if self.noise_type in ["cauchy", "t-student"]:
            # Use Huber loss for heavy-tailed distributions (robust to outliers)
            # Huber loss is less sensitive to outliers than MSE
            def robust_loss(ys_pred, ys):
                return huber_loss(ys_pred, ys, delta=1.35)
            return robust_loss
        elif self.noise_type == "laplace":
            # Laplace noise: use L1-like loss (MAE) which is more robust
            def laplace_loss(ys_pred, ys):
                return torch.abs(ys - ys_pred).mean()
            return laplace_loss
        else:
            # For normal, uniform, and other noise types, use standard MSE
            return mean_squared_error


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(self.dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
class AR1LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, ar_coef=0.5, noise_std=1.0,compute_gradient=False):
        """
        AR(1) Linear Regression: y_t = x_t^T w + epsilon_t
        where epsilon_t = ar_coef * epsilon_{t-1} + u_t, u_t ~ N(0, noise_std^2)
        
        scale: a constant by which to scale the randomly sampled weights
        ar_coef: AR(1) coefficient for error terms
        noise_std: standard deviation of innovation noise
        """
        super(AR1LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.ar_coef = ar_coef
        self.noise_std = noise_std
        self.compute_gradient = compute_gradient
        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        """
        Generate AR(1) linear regression data with correlated errors
        """
        w_b = self.w_b.to(xs_b.device)
        batch_size, n_points, n_dims = xs_b.shape
        
        # Generate linear predictions
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0]
        
        # Generate AR(1) error terms
        ys_ar1 = torch.zeros_like(ys_linear)
        for b in range(batch_size):
            # Generate AR(1) process for errors
            errors = torch.zeros(n_points, device=xs_b.device)
            for t in range(n_points):
                if t == 0:
                    # Initial error
                    errors[t] = torch.randn(1, device=xs_b.device) * self.noise_std
                else:
                    # AR(1) error: epsilon_t = ar_coef * epsilon_{t-1} + u_t
                    errors[t] = self.ar_coef * errors[t-1] + torch.randn(1, device=xs_b.device) * self.noise_std
            
            # Add AR(1) errors to linear predictions
            ys_ar1[b] = ys_linear[b] + errors
        
        return ys_ar1

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class SparseRegressionKiller(Task):
    """
    Case 1: Sparse Regression - "Ridge Trap"
    Prior: Spike-and-Slab (only k=2 dims are non-zero)
    Shows Bayesian advantage over Ridge/OLS
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, k_sparse=2):
        super(SparseRegressionKiller, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.k_sparse = k_sparse
        
        if pool_dict is None and seeds is None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            # Only k_sparse dimensions are non-zero, sampled from Uniform[-1,1]
            for i in range(self.b_size):
                active_dims = torch.randperm(self.n_dims)[:self.k_sparse]
                self.w_b[i, active_dims, 0] = torch.rand(self.k_sparse) * 2 - 1
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                active_dims = torch.randperm(self.n_dims, generator=generator)[:self.k_sparse]
                self.w_b[i, active_dims, 0] = torch.rand(self.k_sparse, generator=generator) * 2 - 1
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, k_sparse=2, **kwargs):
        w = torch.zeros(num_tasks, n_dims, 1)
        for i in range(num_tasks):
            active_dims = torch.randperm(n_dims)[:k_sparse]
            w[i, active_dims, 0] = torch.rand(k_sparse) * 2 - 1
        return {"w": w}
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error


class HeavyTailNoiseKiller(Task):
    """
    Case 2: Heavy-tailed Noise - "OLS Enemy"
    Noise: Student-t with low df (reduced variance) or Cauchy (scaled down)
    Shows robustness of Bayesian vs OLS
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, 
                 noise_type="t-student", df=3.0, noise_scale=0.5):
        super(HeavyTailNoiseKiller, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.noise_type = noise_type
        self.df = df
        self.noise_scale = noise_scale  # Reduced scale for learnable regime
        
        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0]
        
        # Add heavy-tail noise with reduced variance
        if self.noise_type == "t-student":
            noise_dist = torch.distributions.StudentT(df=self.df)
            noise = noise_dist.sample(ys_linear.shape).to(xs_b.device) * self.noise_scale
        elif self.noise_type == "cauchy":
            noise_dist = torch.distributions.Cauchy(loc=0, scale=self.noise_scale)
            noise = noise_dist.sample(ys_linear.shape).to(xs_b.device)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
        
        return ys_linear + noise
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"w": torch.randn(num_tasks, n_dims, 1)}
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        # Use Huber loss for robustness to outliers
        def robust_loss(ys_pred, ys):
            return huber_loss(ys_pred, ys, delta=1.0)
        return robust_loss


class BoundedSupportKiller(Task):
    """
    Case 3: Bounded Support - "Sign Constraint"
    Prior: w ~ Exponential (w > 0 always)
    Input: x ~ Uniform[0, 1] (positive only)
    OLS can predict negative w, Bayes respects constraint
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rate=1.0):
        super(BoundedSupportKiller, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.rate = rate
        
        if pool_dict is None and seeds is None:
            exp_dist = torch.distributions.Exponential(rate=self.rate)
            self.w_b = exp_dist.sample((self.b_size, self.n_dims, 1))
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                exp_dist = torch.distributions.Exponential(rate=self.rate)
                # Manual sampling with generator
                u = torch.rand(self.n_dims, 1, generator=generator)
                self.w_b[i] = -torch.log(u) / self.rate
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, rate=1.0, **kwargs):
        exp_dist = torch.distributions.Exponential(rate=rate)
        return {"w": exp_dist.sample((num_tasks, n_dims, 1))}
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error


class MixtureTasksKiller(Task):
    """
    Case 4: Mixture of Tasks - "Averaging Death"
    Prior: 50% y = w^T x, 50% y = -w^T x
    OLS averages to 0, Bayes maintains bimodal posterior
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(MixtureTasksKiller, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        
        if pool_dict is None and seeds is None:
            # Sample base w
            w_base = torch.randn(self.b_size, self.n_dims, 1)
            # Randomly flip sign for 50% of tasks
            signs = torch.randint(0, 2, (self.b_size, 1, 1)) * 2 - 1  # {-1, +1}
            self.w_b = w_base * signs
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                w_base = torch.randn(self.n_dims, 1, generator=generator)
                sign = torch.randint(0, 2, (1,), generator=generator).item() * 2 - 1
                self.w_b[i] = w_base * sign
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        w_base = torch.randn(num_tasks, n_dims, 1)
        signs = torch.randint(0, 2, (num_tasks, 1, 1)) * 2 - 1
        return {"w": w_base * signs}
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error


class TransferTradeoffTask(Task):
    """
    Case 5: Transfer Tradeoff - p×N experiment (Wakayama)
    Tests Bayes Gap (N) vs Posterior Variance (p)
    Use with different (N, p) configurations
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, 
                 prior_type="mixture_gaussian", mixture_std=2.0):
        super(TransferTradeoffTask, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.prior_type = prior_type
        self.mixture_std = mixture_std
        
        if pool_dict is None and seeds is None:
            if prior_type == "mixture_gaussian":
                # Mixture: 50% N(0,1) + 50% N(0, mixture_std^2)
                mode = torch.randint(0, 2, (self.b_size,))
                self.w_b = torch.randn(self.b_size, self.n_dims, 1)
                self.w_b[mode == 1] *= self.mixture_std
            elif prior_type == "sparse":
                # Sparse prior (like Case 1)
                self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
                k_sparse = max(2, n_dims // 10)
                for i in range(self.b_size):
                    active = torch.randperm(n_dims)[:k_sparse]
                    self.w_b[i, active, 0] = torch.randn(k_sparse)
            else:
                raise ValueError(f"Unknown prior_type: {prior_type}")
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                if prior_type == "mixture_gaussian":
                    mode = torch.randint(0, 2, (1,), generator=generator).item()
                    w = torch.randn(self.n_dims, 1, generator=generator)
                    if mode == 1:
                        w *= self.mixture_std
                    self.w_b[i] = w
                elif prior_type == "sparse":
                    k_sparse = max(2, n_dims // 10)
                    active = torch.randperm(n_dims, generator=generator)[:k_sparse]
                    self.w_b[i, active, 0] = torch.randn(k_sparse, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, prior_type="mixture_gaussian", 
                          mixture_std=2.0, **kwargs):
        if prior_type == "mixture_gaussian":
            mode = torch.randint(0, 2, (num_tasks,))
            w = torch.randn(num_tasks, n_dims, 1)
            w[mode == 1] *= mixture_std
        elif prior_type == "sparse":
            w = torch.zeros(num_tasks, n_dims, 1)
            k_sparse = max(2, n_dims // 10)
            for i in range(num_tasks):
                active = torch.randperm(n_dims)[:k_sparse]
                w[i, active, 0] = torch.randn(k_sparse)
        return {"w": w}
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error

class ScaleMismatchTask(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, train_mode=True):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        if train_mode:
            self.w_b = torch.rand(self.b_size, self.n_dims, 1) * 2 - 1
        else:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1) + 100

        def evaluate(self, xs_b):
            w_b = self.w_b.to(xs_b.device)
            ys_b = (xs_b @ w_b)[:, :, 0]
            return ys_b

        @staticmethod
        def get_metric():
            return squared_error

        @staticmethod
        def get_training_metric():
            return mean_squared_error

class DenseTestKiller(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        # w dense: all dimensions = 0.5
        self.w_b = torch.ones(batch_size, n_dims, 1) * 0.5

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b)[:, :, 0]
        return ys_b
    
    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class MixedTaskKiller(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.w_b = torch.randn(batch_size, n_dims, 1)
        self.is_sin = torch.randint(0, 2, (batch_size,))

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys = xs_b @ w_b[:, :, 0]
        for i in range(self.b_size):
            if self.is_sin[i]:
                ys[i] = torch.sin(ys[i])
        return us

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error 
