
import math

import torch

import itertools


def squared_error(ys_pred, ys):
    ys_pred = ys_pred.squeeze()
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    print(ys.shape)
    ys_pred = ys_pred.squeeze()
    print(ys_pred.shape)

    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()

def l2_error(ys_pred, ys): 
    return (ys - ys_pred).norm(dim=2)

def mean_l2_error(ys_pred, ys):
    return (ys - ys_pred).norm(dim=2).mean()

def normalized_l2_error(ys_pred, ys):
    normalized_ys = ys / ys.norm(dim=2, keepdim=True)
    normalized_ys_pred = ys_pred / ys.norm(dim=2, keepdim=True)
    return (normalized_ys - normalized_ys_pred).norm(dim=2)

def normalized_mean_l2_error(ys_pred, ys):
    normalized_ys = ys / ys.norm(dim=2, keepdim=True)
    normalized_ys_pred = ys_pred / ys.norm(dim=2, keepdim=True)
    return (normalized_ys - normalized_ys_pred).norm(dim=2).mean()

def normalized_squared_l2_error(ys_pred, ys):
    normalized_ys = ys / ys.norm(dim=2, keepdim=True)
    normalized_ys_pred = ys_pred / ys.norm(dim=2, keepdim=True)
    return (normalized_ys - normalized_ys_pred).norm(dim=2).square()

def normalized_mean_squared_l2_error(ys_pred, ys):
    normalized_ys = ys / ys.norm(dim=2, keepdim=True)
    normalized_ys_pred = ys_pred / ys.norm(dim=2, keepdim=True)
    return (normalized_ys - normalized_ys_pred).norm(dim=2).square().mean()

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
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "seq_relu_2nn": RecursiveRelu2nn,
        "seq_linear": RecursiveLinearFunction, 
        "seq_rec_linear": SequentialRecursiveLinearFunction
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

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
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


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
            self.target_tensor = torch.zeros_like(dt_tensor)
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
    
class SlidingWindowSequentialTasks(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        super(SlidingWindowSequentialTasks, self).__init__(n_dims, batch_size, pool_dict, seeds)

    def generate_functions(self, ws):
        """
        return a list of functions that each use w_i,j from ws

        the function should have type (self.b_size, self.n_dims) --> (self.b_size, self.n_dims), as it will be used recursively
        """
        raise NotImplementedError
    
    def generate_sequence(self, x0, sequence_length):
        """
        Generate the sequence starting at x0 of length self.sequence_length, using self.functions and self.sliding_window
        
        x0 : (self.batch_size, self.n_dims)

        Return a batch xs and ys to be used for training.
        """
        xs = torch.zeros((self.b_size, sequence_length, self.n_dims))
        ys = torch.zeros((self.b_size, sequence_length, self.n_dims))
        
        xs[:, 0, :] = x0

        # t = 1
        # f 0 -> i 0
        # t = 2 
        # f 0 -> i 1, f 1 -> i 0
        # t = 4, time 4 in sequence
        # f 0 -> i 3, f 1 -> i 2, f 2  -> i 1,  f 3 -> i 0
        for t in range(1, sequence_length):
            x = torch.sum(torch.stack([f(xs[:, t - (i+1), :]) if t - (i+1) >= 0 else torch.zeros_like(xs[:, 0, :]) for i, f in enumerate(self.functions)]), dim=0)
            # i that was not affected by a function
            x_normalized = x / x.norm(p=2, dim=-1, keepdim=True)
            xs[:, t, :] = x_normalized
            ys[:, t - 1, :] = xs[:, t, :]
        next_x = torch.sum(torch.stack([f(xs[:, sequence_length - (i+1), :]) if sequence_length - (i+1) >= 0 else torch.zeros_like(xs[:, 0, :]) for i, f in enumerate(self.functions)]), dim=0)
        next_x_normalized = next_x / next_x.norm(p=2, dim=-1, keepdim=True)
        ys[:, -1, :] = next_x_normalized

        return xs, ys
    
class RecursiveRelu2nn(SlidingWindowSequentialTasks):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, hidden_layer_size=100):
        super(SlidingWindowSequentialTasks, self).__init__(n_dims, batch_size, pool_dict, seeds)

        self.sliding_window=2
        # self.sequence_length=20
        
        self.scale = scale
        self.hidden_layer_size = n_dims

        dims = [(n_dims, hidden_layer_size), (hidden_layer_size, n_dims)]
        self.ws = {f"w{i},{j}" : torch.randn(dims[j]) for i, j in itertools.product(list(range(self.sliding_window + 1)), (range(2)))}
        self.functions = self.generate_functions()

    def generate_functions(self):
        functions = []
        for i in range(self.sliding_window):
            # functions.append(lambda xs_b: (torch.nn.functional.relu(xs_b @ self.ws[f"w{i},{0}"].to(xs_b.device)) @ self.ws[f"w{i},{1}"].to(xs_b.device)) * math.sqrt(2 / self.hidden_layer_size) * self.scale)
            functions.append(lambda xs_b: (torch.nn.functional.relu(xs_b @ self.ws[f"w{i},{0}"].to(xs_b.device)) @ self.ws[f"w{i},{1}"].to(xs_b.device)) * math.sqrt(2 / self.hidden_layer_size))
        return functions

    @staticmethod 
    def generate_pool_dict(n_dims, num_tasks, sliding_window, hidden_layer_size, **kwargs):
        dims = [(num_tasks, n_dims, hidden_layer_size), (num_tasks, hidden_layer_size, n_dims)]
        return {f"w{i},{j}" : torch.randn(dims[i]) for i, j in zip(range(sliding_window), range(2))}
    
    @staticmethod
    def get_metric():
        return l2_error

    @staticmethod
    def get_training_metric():
        return mean_l2_error
    
class RecursiveLinearFunction(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(RecursiveLinearFunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

        self.n_dims = n_dims
        # self.sequence_length=16
        self.scale = 1 / n_dims

        w = torch.randn((n_dims, n_dims))

        eigenvalues, eigenvectors = torch.linalg.eig(w)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        # NOTE: torch.linalg.eig returns complex eigenvalues, so we need to take the real part
        clamped_eigenvalues = torch.clamp(eigenvalues, max=0.8, min=-0.8)
        clamped_matrix = eigenvectors @ torch.diag(clamped_eigenvalues) @ eigenvectors.t()

        self.w = clamped_matrix

        # self.b = torch.randn((1))
        # self.b = torch.randn((n_dims))

        # print(torch.eig(self.w))

        self.functions = self.generate_functions()

    def generate_sequence(self, x0, sequence_length):
        """
        Generate the sequence starting at x0 of length self.sequence_length.
        At each step, the previous vector is multiplied by the matrix W.

        x0 : (self.batch_size, self.n_dims)
        W : (self.n_dims, self.n_dims) matrix

        Return a batch xs and ys to be used for training.
        """
        W = self.w
        xs = torch.zeros((self.b_size, sequence_length, self.n_dims))
        ys = torch.zeros((self.b_size, sequence_length, self.n_dims))

        # Initialize the first step of the sequence
        xs[:, 0, :] = x0

        # Iterate over the sequence
        for t in range(1, sequence_length):
            # Multiply the previous step by the matrix W
            xs[:, t, :] = torch.matmul(xs[:, t-1, :], W) #+ self.b
            # xs_normalized = xs[:, t, :] / torch.norm(xs[:, t, :], 2, -1, True)
            # Copy the new step to ys
            ys[:, t-1, :] = xs[:, t, :] 

        # Update the last element in ys
        ys[:, sequence_length-1, :] = (torch.matmul(xs[:, -1, :], W)) # + self.b)
        x_means = torch.norm(xs, 2, -1, True)
        return xs, ys

    def generate_functions(self):
        functions = []
        def func(xs):
            w = self.w
            ys = self.scale * torch.stack([w @ xb for  xb in xs])
            return ys
        functions.append(func)
        return functions

    @staticmethod 
    def generate_pool_dict(n_dims, num_tasks, sliding_window, hidden_layer_size, **kwargs):
        dims = [(num_tasks, n_dims, hidden_layer_size), (num_tasks, hidden_layer_size, n_dims)]
        return {f"w{i},{j}" : torch.randn(dims[i]) for i, j in zip(range(sliding_window), range(2))}
    
    @staticmethod
    def get_metric():
        return normalized_l2_error

    @staticmethod
    def get_training_metric():
        return normalized_mean_l2_error

class SequentialRecursiveLinearFunction(SlidingWindowSequentialTasks):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(SlidingWindowSequentialTasks, self).__init__(n_dims, batch_size, pool_dict, seeds)
        #use seq_req_linear.yaml
        self.n_dims = n_dims
        self.scale = 1 / n_dims

        w = torch.randn((n_dims, n_dims))

        eigenvalues, eigenvectors = torch.eig(w, eigenvectors=True)
        clamped_eigenvalues = torch.clamp(eigenvalues[:, 0], max=1.0, min=-1.0)
        clamped_matrix = eigenvectors @ torch.diag(clamped_eigenvalues) @ eigenvectors.t()

        self.w = torch.clamp(clamped_matrix, max=0.5, min=-0.5)
        self.functions = self.generate_functions() 

    def generate_functions(self):
        functions = []
        def func(xs):
            w = self.w
            ys = self.scale * torch.stack([w @ xb for  xb in xs])
            return ys
        functions.append(func)
        return functions

    @staticmethod 
    def generate_pool_dict(n_dims, num_tasks, sliding_window, hidden_layer_size, **kwargs):
        dims = [(num_tasks, n_dims, hidden_layer_size), (num_tasks, hidden_layer_size, n_dims)]
        return {f"w{i},{j}" : torch.randn(dims[i]) for i, j in zip(range(sliding_window), range(2))}
    
    @staticmethod
    def get_metric():
        return normalized_squared_l2_error

    @staticmethod
    def get_training_metric():
        return normalized_mean_squared_l2_error

def get_seq_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "seq_relu_2nn": Relu2nnRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError
    