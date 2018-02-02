## Executing Caffe models using PyTorch as backend
```python
import caffemodel2pytorch

# https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
model = caffemodel2pytorch.Net(prototxt = 'VGG_ILSVRC_16_layers_deploy.prototxt', weights = 'VGG_ILSVRC_16_layers.caffemodel', caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto')
model.cuda()
model.eval()

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

# === loading and using the converted model in evaluation mode
net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', caffe.TEST, weights = 'VGG_ILSVRC_16_layers.caffemodel')

# outputs a dict of NumPy arrays
blobs_out = net.forward(data = np.zeros((8, 3, 512, 512), dtype = np.float32))

# accesses the first layer
layer = net.layers[-1]

# converts and provides the output as NumPy array
numpy_array = net.blobs['conv1_1'].data

# accesse the loss weights
loss_weights = net.blob_loss_weights

# === configuring caffemodel2pytorch to learn about additional layers ===

# setting a path to a custom proto, http paths accepted too
caffe.caffe_proto = 'if_needed_path_to_custom_caffe.proto'

# modules dictionary will be checked when reading the prototxt
# param is a dict representing the layer param, e.g. convolution_param for the Convolution module

# register a nn.Module-derived layer:
caffe.modules['ROIPooling'] = lambda param: CustomRoiPoolingLayer(param['spatial_scale'])

# register a function:
caffe.modules['GlobalSumPooling'] = lambda param: lambda input: torch.sum(input)

# 
solver = caffe.SGDSolver('../oicr/models/VGG16/solver.prototxt', weights = '../oicr/data/imagenet_models/VGG16.v2.caffemodel', train_prototxt = '../oicr/models/VGG16/train.prototxt')
solver.step(1)

```
