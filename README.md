## Executing Caffe models using PyTorch as backend
```python
import caffemodel2pytorch

# https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
model = caffemodel2pytorch.Net(
	prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt',
	weights = 'VGG_ILSVRC_16_layers.caffemodel',
	caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
)
model.cuda()
model.eval()

# make sure to have right procedure of image normalization and channel reordering
image = torch.autograd.Variable(torch.Tensor(8, 3, 512, 512).cuda())

# outputs dict of PyTorch Variables
output_dict = model(data = image)

# a single input variable is interpreted as an input blob named "data"
output_dict = model(image) 

```

## Imitating pycaffe interface to help in porting

```python
import numpy as np
import caffemodel2pytorch as caffe

caffe.set_mode_gpu()
caffe.set_device(0)

# === LOADING AND USING THE NET IN EVAL MODE ===
net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', caffe.TEST, weights = 'VGG_ILSVRC_16_layers.caffemodel')

# outputs a dict of NumPy arrays, data layer is sidestepped
blobs_out = net.forward(data = np.zeros((8, 3, 512, 512), dtype = np.float32))

# accesses the last layer
layer = net.layers[-1]

# converts and provides the output as NumPy array
numpy_array = net.blobs['conv1_1'].data

# accesse the loss weights
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

# accesses the underlying net
solver.net

# load pretrained weights
solver.net.copy_from('oicr/data/imagenet_models/VGG16.v2.caffemodel')

# running one iteration of forward, backward, optimization
# data layer must be provided or data keyword argument provided to step() call
# returns a float loss value
loss = solver.step(1)
```

## Supported layers
* convolution (constant and gaussian weight fillers)
* inner product
* max / avg pooling
* relu
* dropout
* eltwise
* softmax
