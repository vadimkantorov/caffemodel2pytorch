import os
import sys
import time
import argparse
import tempfile
import subprocess
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from urllib.request import urlopen

import google.protobuf.descriptor
import google.protobuf.descriptor_pool
import google.protobuf.symbol_database
import google.protobuf.text_format
import google.protobuf.json_format

import ssl

from layers2d import modules2d
from layers3d import modules3d
from blob import Blob
from utils import to_dict, convert_to_gpu_if_enabled

ssl._create_default_https_context = ssl._create_unverified_context

TRAIN = 0

TEST = 1

caffe_pb2 = None


def initialize(
    caffe_proto="https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto",
    codegen_dir=tempfile.mkdtemp(),
    shadow_caffe=True,
):
    global caffe_pb2
    if caffe_pb2 is None:
        local_caffe_proto = os.path.join(codegen_dir, os.path.basename(caffe_proto))
        with open(local_caffe_proto, "w") as f:
            mybytes = urlopen(caffe_proto).read()
            mystr = mybytes.decode("ascii", "ignore")
            f.write(mystr)
            # f.write((urlopen if 'http' in caffe_proto else open)(caffe_proto).read())
        subprocess.check_call(
            [
                "protoc",
                "--proto_path",
                os.path.dirname(local_caffe_proto),
                "--python_out",
                codegen_dir,
                local_caffe_proto,
            ]
        )
        sys.path.insert(0, codegen_dir)
        old_symdb = google.protobuf.symbol_database._DEFAULT
        google.protobuf.symbol_database._DEFAULT = (
            google.protobuf.symbol_database.SymbolDatabase(
                pool=google.protobuf.descriptor_pool.DescriptorPool()
            )
        )
        import caffe_pb2 as caffe_pb2

        google.protobuf.symbol_database._DEFAULT = old_symdb
        sys.modules[__name__ + ".proto"] = sys.modules[__name__]
        if shadow_caffe:
            sys.modules["caffe"] = sys.modules[__name__]
            sys.modules["caffe.proto"] = sys.modules[__name__]
    return caffe_pb2


def set_mode_gpu():
    global convert_to_gpu_if_enabled
    convert_to_gpu_if_enabled = lambda obj: obj.cuda()


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


class Net(nn.Module):
    def __init__(self, prototxt, *args, **kwargs):
        super().__init__()
        # to account for both constructors, see https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_net.py#L145-L147
        caffe_proto = kwargs.pop("caffe_proto", None)
        weights = kwargs.pop("weights", None)
        print(weights)
        phase = kwargs.pop("phase", None)
        weights = weights or (args + (None, None))[0]
        phase = phase or (args + (None, None))[1]

        self.net_param = initialize(caffe_proto).NetParameter()
        google.protobuf.text_format.Parse(open(prototxt).read(), self.net_param)

        # Check if number_of_inputs is not larger than 1
        number_of_inputs = len(self.net_param.input_shape)
        if number_of_inputs > 1:
            raise NotImplementedError("Multiple inputs is currently not supported.")

        # Check dimensionality
        # The first two dimensions are batch and channel, so subtract 2
        input_dimensions = len(self.net_param.input_shape[0].dim) - 2
        if not input_dimensions in [2, 3]:
            raise NotImplementedError(
                "Only 2D and 3D caffe networks are currently supported."
            )

        # Decide to use 2D or 3D modules
        if input_dimensions == 2:
            print("2D modules selected")
            modules = modules2d
        if input_dimensions == 3:
            print("3D modules selected")
            modules = modules3d

        for layer in list(self.net_param.layer) + list(self.net_param.layers):
            layer_type = (
                layer.type if layer.type != "Python" else layer.python_param.layer
            )

            if isinstance(layer_type, int):
                layer_type = layer.LayerType.Name(layer_type)

            module_constructor = (
                [
                    v
                    for k, v in modules.items()
                    if k.replace("_", "").upper()
                    in [
                        layer_type.replace("_", "").upper(),
                        layer.name.replace("_", "").upper(),
                    ]
                ]
                + [None]
            )[0]

            print(f"Module constructor: {module_constructor}")

            if module_constructor is not None:
                param = to_dict(
                    (
                        [v for f, v in layer.ListFields() if f.name.endswith("_param")]
                        + [None]
                    )[0]
                )
                caffe_input_variable_names = list(layer.bottom)
                caffe_output_variable_names = list(layer.top)
                caffe_loss_weight = (
                    list(layer.loss_weight)
                    or [1.0 if layer_type.upper().endswith("LOSS") else 0.0]
                ) * len(layer.top)
                caffe_propagate_down = list(getattr(layer, "propagate_down", [])) or [
                    True
                ] * len(caffe_input_variable_names)
                caffe_optimization_params = to_dict(layer.param)
                param["inplace"] = (
                    len(caffe_input_variable_names) == 1
                    and caffe_input_variable_names == caffe_output_variable_names
                )
                module = module_constructor(param)
                self.add_module(
                    layer.name,
                    module
                    if isinstance(module, nn.Module)
                    else CaffePythonLayerModule(
                        module,
                        caffe_input_variable_names,
                        caffe_output_variable_names,
                        param.get("param_str", ""),
                    )
                    if type(module).__name__.endswith("Layer")
                    else FunctionModule(module),
                )
                module = getattr(self, layer.name)
                module.caffe_layer_name = layer.name
                module.caffe_layer_type = layer_type
                module.caffe_input_variable_names = caffe_input_variable_names
                module.caffe_output_variable_names = caffe_output_variable_names
                module.caffe_loss_weight = caffe_loss_weight
                module.caffe_propagate_down = caffe_propagate_down
                module.caffe_optimization_params = caffe_optimization_params
                for optim_param, param in zip(
                    caffe_optimization_params, module.parameters()
                ):
                    param.requires_grad = optim_param.get("lr_mult", 1) != 0
            else:
                print(
                    (
                        f"Skipping layer [{layer.name}, {layer_type}, {layer.type}]:"
                        "not found in caffemodel2pytorch.modules dict"
                    )
                )

        if weights is not None:
            self.copy_from(weights)

        self.blobs = collections.defaultdict(Blob)
        self.blob_loss_weights = {
            name: loss_weight
            for module in self.children()
            for name, loss_weight in zip(
                module.caffe_output_variable_names, module.caffe_loss_weight
            )
        }

        self.train(phase != TEST)
        convert_to_gpu_if_enabled(self)

    def forward(self, data=None, **variables):
        if data is not None:
            variables["data"] = data
        numpy = not all(map(torch.is_tensor, variables.values()))
        variables = {
            k: convert_to_gpu_if_enabled(torch.from_numpy(v.copy()) if numpy else v)
            for k, v in variables.items()
        }

        for module in [
            module
            for module in self.children()
            if not all(name in variables for name in module.caffe_output_variable_names)
        ]:
            for name in module.caffe_input_variable_names:
                assert name in variables, (
                    f"Variable [{name}] does not exist. "
                    "Pass it as a keyword argument or provide a layer which produces it."
                )
            inputs = [
                variables[name] if propagate_down else variables[name].detach()
                for name, propagate_down in zip(
                    module.caffe_input_variable_names, module.caffe_propagate_down
                )
            ]
            outputs = module(*inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            variables.update(dict(zip(module.caffe_output_variable_names, outputs)))

        self.blobs.update({k: Blob(data=v, numpy=numpy) for k, v in variables.items()})
        caffe_output_variable_names = set(
            [
                name
                for module in self.children()
                for name in module.caffe_output_variable_names
            ]
        ) - set(
            [
                name
                for module in self.children()
                for name in module.caffe_input_variable_names
                if name not in module.caffe_output_variable_names
            ]
        )
        return {
            k: v.detach().cpu().numpy() if numpy else v
            for k, v in variables.items()
            if k in caffe_output_variable_names
        }

    def copy_from(self, weights):
        try:
            import h5py, numpy

            state_dict = self.state_dict()
            for k, v in h5py.File(weights, "r").items():
                if k in state_dict:
                    state_dict[k].resize_(v.shape).copy_(
                        torch.from_numpy(numpy.array(v))
                    )
            print(
                "caffemodel2pytorch: loaded model from [{}] in HDF5 format".format(
                    weights
                )
            )
        except Exception as e:
            print(
                "caffemodel2pytorch: loading model from [{}] in HDF5 format failed [{}], falling back to caffemodel format".format(
                    weights, e
                )
            )
            bytes_weights = open(weights, "rb").read()
            bytes_parsed = self.net_param.ParseFromString(bytes_weights)
            if bytes_parsed != len(bytes_weights):
                print(
                    "caffemodel2pytorch: loading model from [{}] in caffemodel format, WARNING: file length [{}] is not equal to number of parsed bytes [{}]".format(
                        weights, len(bytes_weights), bytes_parsed
                    )
                )
            for layer in list(self.net_param.layer) + list(self.net_param.layers):
                module = getattr(self, layer.name, None)
                if module is None:
                    continue
                parameters = {
                    name: convert_to_gpu_if_enabled(torch.FloatTensor(blob.data)).view(
                        list(blob.shape.dim)
                        if len(blob.shape.dim) > 0
                        else [blob.num, blob.channels, blob.height, blob.width]
                    )
                    for name, blob in zip(["weight", "bias"], layer.blobs)
                }
                if len(parameters) > 0:
                    print(f"Weights shape: {parameters['weight'].shape}")
                    print(f"Bias: {parameters['bias']}")
                    module.set_parameters(**parameters)
            print(
                "caffemodel2pytorch: loaded model from [{}] in caffemodel format".format(
                    weights
                )
            )

    def save(self, weights):
        import h5py

        with h5py.File(weights, "w") as h:
            for k, v in self.state_dict().items():
                h[k] = v.cpu().numpy()
        print("caffemodel2pytorch: saved model to [{}] in HDF5 format".format(weights))

    @property
    def layers(self):
        return list(self.children())


class FunctionModule(nn.Module):
    def __init__(self, forward):
        super(FunctionModule, self).__init__()
        self.forward_func = forward

    def forward(self, *inputs):
        return self.forward_func(*inputs)


class CaffePythonLayerModule(nn.Module):
    def __init__(
        self,
        caffe_python_layer,
        caffe_input_variable_names,
        caffe_output_variable_names,
        param_str,
    ):
        super(CaffePythonLayerModule, self).__init__()
        caffe_python_layer.param_str = param_str
        self.caffe_python_layer = caffe_python_layer
        self.caffe_input_variable_names = caffe_input_variable_names
        self.caffe_output_variable_names = caffe_output_variable_names

    def forward(self, *inputs):
        return Layer(
            self.caffe_python_layer,
            self.caffe_input_variable_names,
            self.caffe_output_variable_names,
        )(*inputs)

    def __getattr__(self, name):
        return (
            nn.Module.__getattr__(self, name)
            if name in dir(self)
            else getattr(self.caffe_python_layer, name)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        metavar="model.caffemodel",
        dest="model_caffemodel",
        help="Path to model.caffemodel",
    )
    parser.add_argument(
        "-o",
        dest="output_path",
        help="Path to converted model, supported file extensions are: h5, npy, npz, json, pt",
    )
    parser.add_argument(
        "--caffe.proto",
        metavar="--caffe.proto",
        dest="caffe_proto",
        help="Path to caffe.proto (typically located at CAFFE_ROOT/src/caffe/proto/caffe.proto)",
        default="https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto",
    )
    args = parser.parse_args()
    args.output_path = args.output_path or args.model_caffemodel + ".pt"

    net_param = initialize(args.caffe_proto).NetParameter()
    net_param.ParseFromString(open(args.model_caffemodel, "rb").read())
    blobs = {
        layer.name
        + "."
        + name: dict(
            data=blob.data,
            shape=list(blob.shape.dim)
            if len(blob.shape.dim) > 0
            else [blob.num, blob.channels, blob.height, blob.width],
        )
        for layer in list(net_param.layer) + list(net_param.layers)
        for name, blob in zip(["weight", "bias"], layer.blobs)
    }

    if args.output_path.endswith(".json"):
        import json

        with open(args.output_path, "w") as f:
            json.dump(blobs, f)
    elif args.output_path.endswith(".h5"):
        import h5py, numpy

        with h5py.File(args.output_path, "w") as h:
            h.update(
                **{
                    k: numpy.array(blob["data"], dtype=numpy.float32).reshape(
                        *blob["shape"]
                    )
                    for k, blob in blobs.items()
                }
            )
    elif args.output_path.endswith(".npy") or args.output_path.endswith(".npz"):
        import numpy

        (numpy.savez if args.output_path[-1] == "z" else numpy.save)(
            args.output_path,
            **{
                k: numpy.array(blob["data"], dtype=numpy.float32).reshape(
                    *blob["shape"]
                )
                for k, blob in blobs.items()
            },
        )
    elif args.output_path.endswith(".pt"):
        torch.save(
            {
                k: torch.FloatTensor(blob["data"]).view(*blob["shape"])
                for k, blob in blobs.items()
            },
            args.output_path,
        )
