from torch import nn
import collections

from google.protobuf.descriptor import FieldDescriptor as FD


def init_weight_bias(self, weight=None, bias=None, requires_grad=[]):
    if weight is not None:
        self.weight = nn.Parameter(
            weight.type_as(self.weight), requires_grad=self.weight.requires_grad
        )
    if bias is not None:
        self.bias = nn.Parameter(
            bias.type_as(self.bias), requires_grad=self.bias.requires_grad
        )
    for name, requires_grad in zip(["weight", "bias"], requires_grad):
        param, init = getattr(self, name), getattr(self, name + "_init")
        if init.get("type") == "gaussian":
            nn.init.normal_(param, std=init["std"])
        elif init.get("type") == "constant":
            nn.init.constant_(param, val=init["value"])
        param.requires_grad = requires_grad


def convert_to_gpu_if_enabled(obj):
    return obj


def first_or(param, key, default):
    val = param.get(key, None)
    print(f"{key}, {val}, {default}")
    if not val:
        return default
    if isinstance(val, float):
        return val
    return val


def to_dict(obj):
    return (
        list(map(to_dict, obj))
        if isinstance(obj, collections.Iterable)
        else {}
        if obj is None
        else {
            f.name: converter(v)
            if f.label != FD.LABEL_REPEATED
            else list(map(converter, v))
            for f, v in obj.ListFields()
            for converter in [
                {
                    FD.TYPE_DOUBLE: float,
                    FD.TYPE_SFIXED32: float,
                    FD.TYPE_SFIXED64: float,
                    FD.TYPE_SINT32: int,
                    FD.TYPE_SINT64: int,
                    FD.TYPE_FLOAT: float,
                    FD.TYPE_ENUM: int,
                    FD.TYPE_UINT32: int,
                    FD.TYPE_INT64: int,
                    FD.TYPE_UINT64: int,
                    FD.TYPE_INT32: int,
                    FD.TYPE_FIXED64: float,
                    FD.TYPE_FIXED32: float,
                    FD.TYPE_BOOL: bool,
                    FD.TYPE_STRING: str,
                    FD.TYPE_BYTES: lambda x: x.encode("string_escape"),
                    FD.TYPE_MESSAGE: to_dict,
                }[f.type]
            ]
        }
    )
