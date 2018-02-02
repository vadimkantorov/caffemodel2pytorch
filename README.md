## Run Caffe models using PyTorch as backend
```python
import torch
import caffemodel2pytorch

# prototxt and caffemodel from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
model = caffemodel2pytorch.Net(
	prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt',
	weights = 'VGG_ILSVRC_16_layers.caffemodel',
	caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
)
model.cuda()
model.eval()
torch.set_grad_enabled(False)

# make sure to have right procedure of image normalization and channel reordering
image = torch.autograd.Variable(torch.Tensor(8, 3, 224, 224).cuda())

# outputs dict of PyTorch Variables
# in this example the dict contains the only key "prob"
#output_dict = model(data = image)

# you can remove unneeded layers:
del model.prob
del model.fc8

# a single input variable is interpreted as an input blob named "data"
# in this example the dict contains the only key "fc7"
output_dict = model(image)
```

## Imitate pycaffe interface to help in porting

```python
import numpy as np
import caffemodel2pytorch as caffe

caffe.set_mode_gpu()
caffe.set_device(0)

# === LOADING AND USING THE NET IN EVAL MODE ===

net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', caffe.TEST, weights = 'VGG_ILSVRC_16_layers.caffemodel')

# outputs a dict of NumPy arrays, data layer is sidestepped
blobs_out = net.forward(data = np.zeros((8, 3, 224, 224), dtype = np.float32))

# access the last layer
layer = net.layers[-1]

# converts and provides the output as NumPy array
numpy_array = net.blobs['conv1_1'].data

# access the loss weights
loss_weights = net.blob_loss_weights

# === CONFIGURING ADDITIONAL LAYERS ===

# setting a path to a custom proto, http paths accepted too
caffe.caffe_proto = 'if_needed_path_to_custom_caffe.proto'

# modules dictionary will be checked when reading the prototxt (by layer type or by layer name, case invariant)
# param is a dict representing the layer param, e.g. convolution_param for the Convolution module

# register a nn.Module-derived layer
caffe.modules['ROIPooling'] = lambda param: CustomRoiPoolingLayer(param['spatial_scale'])

# register a function
caffe.modules['GlobalSumLayer'] = lambda param: lambda input: torch.sum(input)

# register a data layer
caffe.modules['data'] = lambda param: lambda *args: torch.cuda.FloatTensor(8, 3, 512, 512)

# === BASIC OPTIMIZER ===

# this example uses paths from https://github.com/ppengtang/oicr

# create an SGD solver, loads the net in train mode
# it knows about base_lr, weight_decay, momentum, lr_mult, decay_mult, iter_size, lr policy step, step_size, gamma
# it finds train.prototxt from the solver.prototxt's train_net or net parameters
solver = caffe.SGDSolver('oicr/models/VGG16/solver.prototxt')

# access the underlying net
solver.net

# load pretrained weights
solver.net.copy_from('oicr/data/imagenet_models/VGG16.v2.caffemodel')

# runs one iteration of forward, backward, optimization; returns a float loss value
# data layer must be registered or inputs must be provided as keyword arguments
loss = solver.step(1)
```

## Supported layers
* convolution (num_output, kernel_size, stride, pad, dilation; constant and gaussian weight/bias fillers)
* inner_product (num_output; constant and gaussian weight/bias fillers)
* max / avg pooling (kernel_size, stride, pad)
* relu
* dropout (dropout_ratio)
* eltwise (prod, sum, max)
* softmax (axis)
