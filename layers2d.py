import torch
from torch import nn

from blob import Blob
from utils import first_or, init_weight_bias


class Layer(torch.autograd.Function):
    def __init__(
        self,
        caffe_python_layer=None,
        caffe_input_variable_names=None,
        caffe_output_variable_names=None,
        caffe_propagate_down=None,
    ):
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names
        self.caffe_propagate_down = caffe_propagate_down

    def forward(self, *inputs):
        bottom = [Blob(data=v.cpu().numpy()) for v in inputs]
        top = [Blob() for name in self.caffe_output_variable_names]

        # self.caffe_python_layer.reshape()
        self.caffe_python_layer.setup(bottom, top)
        self.caffe_python_layer.setup = lambda *args: None

        self.caffe_python_layer.forward(bottom, top)
        outputs = tuple(
            convert_to_gpu_if_enabled(
                torch.from_numpy(v.data.contents.reshape(*v.shape))
            )
            for v in top
        )
        self.save_for_backward(*(inputs + outputs))
        return outputs

    def backward(self, grad_outputs):
        inputs, outputs = (
            self.saved_tensors[: len(self.caffe_input_variable_names)],
            self.saved_tensors[len(self.caffe_input_variable_names) :],
        )
        bottom = [Blob(data=v.cpu().numpy()) for v in inputs]
        top = [
            Blob(data=output.cpu().numpy(), diff=grad_output.cpu().numpy())
            for grad_output, output in zip(grad_outputs, outputs)
        ]
        self.caffe_python_layer.backward(top, self.caffe_propagate_down, bottom)
        return tuple(
            convert_to_gpu_if_enabled(
                torch.from_numpy(blob.diff.contents.reshape(*v.reshape))
            )
            if propagate_down
            else None
            for v, propagate_down in zip(bottom, self.caffe_propagate_down)
        )


class SGDSolver(object):
    def __init__(self, solver_prototxt):
        solver_param = initialize().SolverParameter()
        google.protobuf.text_format.Parse(open(solver_prototxt).read(), solver_param)
        solver_param = to_dict(solver_param)
        self.net = Net(
            solver_param.get("train_net") or solver_param.get("net"), phase=TRAIN
        )
        self.iter = 1
        self.iter_size = solver_param.get("iter_size", 1)
        self.optimizer_params = dict(
            lr=solver_param.get("base_lr") / self.iter_size,
            momentum=solver_param.get("momentum", 0),
            weight_decay=solver_param.get("weight_decay", 0),
        )
        self.lr_scheduler_params = dict(
            policy=solver_param.get("lr_policy"),
            step_size=solver_param.get("stepsize"),
            gamma=solver_param.get("gamma"),
        )
        self.optimizer, self.scheduler = None, None

    def init_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD(
            [
                dict(
                    params=[param],
                    lr=self.optimizer_params["lr"] * mult.get("lr_mult", 1),
                    weight_decay=self.optimizer_params["weight_decay"]
                    * mult.get("decay_mult", 1),
                    momentum=self.optimizer_params["momentum"],
                )
                for module in self.net.children()
                for param, mult in zip(
                    module.parameters(), module.caffe_optimization_params + [{}, {}]
                )
                if param.requires_grad
            ]
        )
        self.scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_scheduler_params["step_size"],
                gamma=self.lr_scheduler_params["gamma"],
            )
            if self.lr_scheduler_params.get("policy") == "step"
            else type("", (object,), dict(step=lambda self: None))()
        )

    def step(self, iterations=1, **inputs):
        loss_total = 0.0
        for i in range(iterations):
            tic = time.time()
            if self.optimizer is not None:
                self.optimizer.zero_grad()

            loss_batch = 0
            losses_batch = collections.defaultdict(float)
            for j in range(self.iter_size):
                outputs = [
                    kv
                    for kv in self.net(**inputs).items()
                    if self.net.blob_loss_weights[kv[0]] != 0
                ]
                loss = sum(
                    [self.net.blob_loss_weights[k] * v.sum() for k, v in outputs]
                )
                loss_batch += float(loss) / self.iter_size
                for k, v in outputs:
                    losses_batch[k] += float(v.sum()) / self.iter_size
                if self.optimizer is None:
                    self.init_optimizer_scheduler()
                    self.optimizer.zero_grad()
                loss.backward()

            loss_total += loss_batch
            self.optimizer.step()
            self.scheduler.step()
            self.iter += 1

            log_prefix = self.__module__ + "." + type(self).__name__
            print(
                "{}] Iteration {}, loss: {}".format(log_prefix, self.iter, loss_batch)
            )
            for i, (name, loss) in enumerate(sorted(losses_batch.items())):
                print(
                    "{}]     Train net output #{}: {} = {} (* {} = {} loss)".format(
                        log_prefix,
                        i,
                        name,
                        loss,
                        self.net.blob_loss_weights[name],
                        self.net.blob_loss_weights[name] * loss,
                    )
                )
            print(
                "{}] Iteration {}, lr = {}, time = {}".format(
                    log_prefix,
                    self.iter,
                    self.optimizer_params["lr"],
                    time.time() - tic,
                )
            )

        return loss_total


class Convolution(nn.Conv2d):
    def __init__(self, param):
        super(Convolution, self).__init__(
            first_or(param, "group", 1),
            param["num_output"],
            kernel_size=first_or(param, "kernel_size", 1),
            stride=first_or(param, "stride", 1),
            padding=first_or(param, "pad", 0),
            dilation=first_or(param, "dilation", 1),
            groups=first_or(param, "group", 1),
        )
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get("weight_filler", {}), param.get(
            "bias_filler", {}
        )

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad]
            super(Convolution, self).__init__(
                x.size(1),
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(Convolution, self).forward(x)

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(
            self, weight=weight, bias=bias.view(-1) if bias is not None else bias
        )
        self.in_channels = self.weight.size(1)


class Deconvolution(nn.ConvTranspose2d):
    def __init__(self, param):
        super(Deconvolution, self).__init__(
            first_or(param, "group", 1),
            param["num_output"],
            kernel_size=first_or(param, "kernel_size", 1),
            stride=first_or(param, "stride", 1),
            padding=first_or(param, "pad", 0),
            dilation=first_or(param, "dilation", 1),
            groups=first_or(param, "group", 1),
        )
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get("weight_filler", {}), param.get(
            "bias_filler", {}
        )

    def forward(self, x):
        if self.weight_numel() == 0 and self.bias_numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad]
            super(Deconvolution, self).__init__(
                x.size(1),
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(Deconvolution, self).forward(x)

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(
            self, weight=weight, bias=bias.view(-1) if bias is not None else bias
        )
        self.in_channels = self.weight.size(1)


class InnerProduct(nn.Linear):
    def __init__(self, param):
        super(InnerProduct, self).__init__(1, param["num_output"])
        self.weight, self.bias = nn.Parameter(), nn.Parameter()
        self.weight_init, self.bias_init = param.get("weight_filler", {}), param.get(
            "bias_filler", {}
        )

    def forward(self, x):
        if self.weight.numel() == 0 and self.bias.numel() == 0:
            requires_grad = [self.weight.requires_grad, self.bias.requires_grad]
            super(InnerProduct, self).__init__(x.size(1), self.out_features)
            convert_to_gpu_if_enabled(self)
            init_weight_bias(self, requires_grad=requires_grad)
        return super(InnerProduct, self).forward(
            x if x.size(-1) == self.in_features else x.view(len(x), -1)
        )

    def set_parameters(self, weight=None, bias=None):
        init_weight_bias(
            self,
            weight=weight.view(weight.size(-2), weight.size(-1))
            if weight is not None
            else None,
            bias=bias.view(-1) if bias is not None else None,
        )
        self.in_features = self.weight.size(1)


# using dict calls (=) otherwise the dict does not play nice with
# the lambda function
modules2d = dict(
    Convolution=lambda param: Convolution(param),
    Deconvolution=lambda param: Deconvolution(param),
    InnerProduct=lambda param: InnerProduct(param),
    Pooling=lambda param: [nn.MaxPool2d, nn.AvgPool2d][param["pool"]](
        kernel_size=first_or(param, "kernel_size", 1),
        stride=first_or(param, "stride", 1),
        padding=first_or(param, "pad", 0),
    ),
    Softmax=lambda param: nn.Softmax(dim=param.get("axis", -1)),
    ReLU=lambda param: nn.ReLU(),
    Dropout=lambda param: nn.Dropout(p=param["dropout_ratio"]),
    Eltwise=lambda param: [torch.mul, torch.add, torch.max][param.get("operation", 1)],
    Concat=lambda param: torch.cat,
    LRN=lambda param: nn.LocalResponseNorm(
        size=param["local_size"], alpha=param["alpha"], beta=param["beta"]
    ),
)
