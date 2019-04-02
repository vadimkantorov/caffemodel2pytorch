This converter can be useful for porting Caffe code and layers to PyTorch. Features:
* dump caffemodel weights to hdf5, npy, pt and json formats
* load Caffe models and use them from PyTorch
* mock PyCaffe API to allow for smooth porting of Caffe-using code (drop-in script for [OICR](https://github.com/ppengtang/oicr) for changing backend in train/eval to PyTorch is below):
  * Net, Blob, SGDSolver
* wrapping Caffe's Python layers (see the OICR example)
* example of ROI pooling in PyTorch without manual CUDA code compilation (see the OICR example)

The layer support isn't as complete as in https://github.com/marvis/pytorch-caffe. Currently it supports the following Caffe layers:
* convolution (num_output, kernel_size, stride, pad, dilation; constant and gaussian weight/bias fillers)
* inner_product (num_output; constant and gaussian weight/bias fillers)
* max / avg pooling (kernel_size, stride, pad)
* relu
* dropout (dropout_ratio)
* eltwise (prod, sum, max)
* softmax (axis)
* local response norm (local_size, alpha, beta)

Dependencies: protobuf with Python bindings, including `protoc` binary in `PATH`.

PRs to enable other layers or layer params are very welcome (see the definition of the `modules` dictionary in the code)!

License is MIT.

## Dump weights to PT or HDF5
```shell
# prototxt and caffemodel from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

# dumps to PT by default to VGG_ILSVRC_16_layers.caffemodel.pt
python -m caffemodel2pytorch VGG_ILSVRC_16_layers.caffemodel

# dumps to HDF5 converted.h5
python -m caffemodel2pytorch VGG_ILSVRC_16_layers.caffemodel -o converted.h5
```

```python
# load dumped VGG16 in PyTorch
import collections, torch, torchvision, numpy, h5py
model = torchvision.models.vgg16()
model.features = torch.nn.Sequential(collections.OrderedDict(zip(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], model.features)))
model.classifier = torch.nn.Sequential(collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8'], model.classifier)))

state_dict = h5py.File('converted.h5', 'r') # torch.load('VGG_ILSVRC_16_layers.caffemodel.pt')
model.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model.named_parameters() if k in l})
```

## Run Caffe models using PyTorch as backend
```python
import torch
import caffemodel2pytorch

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

# === BASIC OPTIMIZER ===

# this example uses paths from https://github.com/ppengtang/oicr

# create an SGD solver, loads the net in train mode
# it knows about base_lr, weight_decay, momentum, lr_mult, decay_mult, iter_size, lr policy step, step_size, gamma
# it finds train.prototxt from the solver.prototxt's train_net or net parameters
solver = caffe.SGDSolver('oicr/models/VGG16/solver.prototxt')

# load pretrained weights
solver.net.copy_from('oicr/data/imagenet_models/VGG16.v2.caffemodel')

# runs one iteration of forward, backward, optimization; returns a float loss value
# data layer must be registered or inputs must be provided as keyword arguments
loss = solver.step(1)
```

## Drop-in script for OICR enabling PyTorch as backend for eval and training
Place `caffe_pytorch_oicr.py` and `caffemodel2pytorch.py` in the root `oicr` directory. To use the PyTorch backend in testing and in training, put a line `import caffe_pytorch_oicr` at the very top (before `import _init_paths`) in `tools/test_net.py` and `tools/train_net.py` respectively. It requires PyTorch and CuPy (for on-the-fly CUDA kernel compilation).

```python
# caffe_pytorch_oicr.py

import collections
import torch
import torch.nn.functional as F
import cupy
import caffemodel2pytorch

caffemodel2pytorch.initialize('./caffe-oicr/src/caffe/proto/caffe.proto') # needs to be called explicitly for these porting scenarios to enable caffe.proto.caffe_pb2 variable
caffemodel2pytorch.set_mode_gpu()
caffemodel2pytorch.modules['GlobalSumPooling'] = lambda param: lambda pred: pred.sum(dim = 0, keepdim = True)
caffemodel2pytorch.modules['MulticlassCrossEntropyLoss'] = lambda param: lambda pred, labels, eps = 1e-6: F.binary_cross_entropy(pred.clamp(eps, 1 - eps), labels)
caffemodel2pytorch.modules['data'] = lambda param: __import__('roi_data_layer.layer').layer.RoIDataLayer() # wrapping a PyCaffe layer
caffemodel2pytorch.modules['OICRLayer'] = lambda param: OICRLayer # wrapping a PyTorch function
caffemodel2pytorch.modules['WeightedSoftmaxWithLoss'] = lambda param: WeightedSoftmaxWithLoss
caffemodel2pytorch.modules['ReLU'] = lambda param: torch.nn.ReLU(inplace = True) # wrapping a PyTorch module
caffemodel2pytorch.modules['ROIPooling'] = lambda param: lambda input, rois: RoiPooling(param['pooled_h'], param['pooled_w'], param['spatial_scale'])(input, rois) # wrapping a PyTorch autograd function

def WeightedSoftmaxWithLoss(prob, labels_ic, cls_loss_weights, eps = 1e-12):
	loss = -cls_loss_weights * F.log_softmax(prob, dim = -1).gather(-1, labels_ic.long().unsqueeze(-1)).squeeze(-1)
	valid_sum = cls_loss_weights.gt(eps).float().sum()
	return loss.sum() / (loss.numel() if valid_sum == 0 else valid_sum)

def OICRLayer(boxes, cls_prob, im_labels, cfg_TRAIN_FG_THRESH = 0.5):
    cls_prob = (cls_prob if cls_prob.size(-1) == im_labels.size(-1) else cls_prob[..., 1:]).clone()
    boxes = boxes[..., 1:]
    gt_boxes, gt_classes, gt_scores = [], [], []
    for i in im_labels.eq(1).nonzero()[:, 1]:
        max_index = int(cls_prob[:, i].max(dim = 0)[1])
        gt_boxes.append(boxes[max_index])
        gt_classes.append(int(i) + 1)
        gt_scores.append(float(cls_prob[max_index, i]))
        cls_prob[max_index] = 0
    max_overlaps, gt_assignment = overlap(boxes, torch.stack(gt_boxes)).max(dim = 1)
    return gt_assignment.new(gt_classes)[gt_assignment] * (max_overlaps > cfg_TRAIN_FG_THRESH).type_as(gt_assignment), max_overlaps.new(gt_scores)[gt_assignment]

class RoiPooling(torch.autograd.Function):
	CUDA_NUM_THREADS = 1024
	GET_BLOCKS = staticmethod(lambda N: (N + RoiPooling.CUDA_NUM_THREADS - 1) // RoiPooling.CUDA_NUM_THREADS)
	Stream = collections.namedtuple('Stream', ['ptr'])

	kernel_forward = b'''
	#define FLT_MAX 340282346638528859811704183484516925440.0f
	typedef float Dtype;
	#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	extern "C"
	__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,
		const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
	  CUDA_KERNEL_LOOP(index, nthreads) { 
		// (n, c, ph, pw) is an element in the pooled output
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		bottom_rois += n * 5;
		int roi_batch_ind = bottom_rois[0];
		int roi_start_w = round(bottom_rois[1] * spatial_scale);
		int roi_start_h = round(bottom_rois[2] * spatial_scale);
		int roi_end_w = round(bottom_rois[3] * spatial_scale);
		int roi_end_h = round(bottom_rois[4] * spatial_scale);

		// Force malformed ROIs to be 1x1
		int roi_width = max(roi_end_w - roi_start_w + 1, 1);
		int roi_height = max(roi_end_h - roi_start_h + 1, 1);
		Dtype bin_size_h = static_cast<Dtype>(roi_height)
						   / static_cast<Dtype>(pooled_height);
		Dtype bin_size_w = static_cast<Dtype>(roi_width)
						   / static_cast<Dtype>(pooled_width);

		int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
											* bin_size_h));
		int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
											* bin_size_w));
		int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
										 * bin_size_h));
		int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
										 * bin_size_w));

		// Add roi offsets and clip to input boundaries
		hstart = min(max(hstart + roi_start_h, 0), height);
		hend = min(max(hend + roi_start_h, 0), height);
		wstart = min(max(wstart + roi_start_w, 0), width);
		wend = min(max(wend + roi_start_w, 0), width);
		bool is_empty = (hend <= hstart) || (wend <= wstart);

		// Define an empty pooling region to be zero
		Dtype maxval = is_empty ? 0 : -FLT_MAX;
		// If nothing is pooled, argmax = -1 causes nothing to be backprop'd
		int maxidx = -1;
		bottom_data += (roi_batch_ind * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
		  for (int w = wstart; w < wend; ++w) {
			int bottom_index = h * width + w;
			if (bottom_data[bottom_index] > maxval) {
			  maxval = bottom_data[bottom_index];
			  maxidx = bottom_index;
			}
		  }
		}
		top_data[index] = maxval;
		argmax_data[index] = maxidx;
	  }
	}
	'''

	kernel_backward = b'''
	typedef float Dtype;
	#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
	extern "C"
	__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
		const int* argmax_data, const int num_rois, const Dtype spatial_scale,
		const int channels, const int height, const int width,
		const int pooled_height, const int pooled_width, Dtype* bottom_diff,
		const Dtype* bottom_rois) {
	  CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;

		Dtype gradient = 0;
		// Accumulate gradient over all ROIs that pooled this element
		for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
		  const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
		  int roi_batch_ind = offset_bottom_rois[0];
		  // Skip if ROI's batch index doesn't match n
		  if (n != roi_batch_ind) {
			continue;
		  }

		  int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
		  int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
		  int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
		  int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

		  // Skip if ROI doesn't include (h, w)
		  const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
							   h >= roi_start_h && h <= roi_end_h);
		  if (!in_roi) {
			continue;
		  }

		  int offset = (roi_n * channels + c) * pooled_height * pooled_width;
		  const Dtype* offset_top_diff = top_diff + offset;
		  const int* offset_argmax_data = argmax_data + offset;

		  // Compute feasible set of pooled units that could have pooled
		  // this bottom unit

		  // Force malformed ROIs to be 1x1
		  int roi_width = max(roi_end_w - roi_start_w + 1, 1);
		  int roi_height = max(roi_end_h - roi_start_h + 1, 1);

		  Dtype bin_size_h = static_cast<Dtype>(roi_height)
							 / static_cast<Dtype>(pooled_height);
		  Dtype bin_size_w = static_cast<Dtype>(roi_width)
							 / static_cast<Dtype>(pooled_width);

		  int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
		  int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
		  int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
		  int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

		  phstart = min(max(phstart, 0), pooled_height);
		  phend = min(max(phend, 0), pooled_height);
		  pwstart = min(max(pwstart, 0), pooled_width);
		  pwend = min(max(pwend, 0), pooled_width);

		  for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
			  if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
				gradient += offset_top_diff[ph * pooled_width + pw];
			  }
			}
		  }
		}
		bottom_diff[index] = gradient;
	  }
	}
	'''
	cupy_init = cupy.array([])
	compiled_forward = cupy.cuda.compiler.compile_with_cache(kernel_forward).get_function('ROIPoolForward')
	compiled_backward = cupy.cuda.compiler.compile_with_cache(kernel_backward).get_function('ROIPoolBackward')

	def __init__(self, pooled_height, pooled_width, spatial_scale):
		self.pooled_height = pooled_height
		self.pooled_width = pooled_width
		self.spatial_scale = spatial_scale

	def forward(self, images, rois):
		output = torch.cuda.FloatTensor(len(rois), images.size(1) * self.pooled_height * self.pooled_width)
		self.argmax = torch.cuda.IntTensor(output.size()).fill_(-1)
		self.input_size = images.size()
		self.save_for_backward(rois)
		RoiPooling.compiled_forward(grid = (RoiPooling.GET_BLOCKS(output.numel()), 1, 1), block = (RoiPooling.CUDA_NUM_THREADS, 1, 1), args=[
			output.numel(), images.data_ptr(), cupy.float32(self.spatial_scale), self.input_size[-3], self.input_size[-2], self.input_size[-1],
			self.pooled_height, self.pooled_width, rois.data_ptr(), output.data_ptr(), self.argmax.data_ptr()
			  ], stream=RoiPooling.Stream(ptr=torch.cuda.current_stream().cuda_stream))
		return output

	def backward(self, grad_output):
		rois, = self.saved_tensors
		grad_input = torch.cuda.FloatTensor(*self.input_size).zero_()
		RoiPooling.compiled_backward(grid = (RoiPooling.GET_BLOCKS(grad_input.numel()), 1, 1), block = (RoiPooling.CUDA_NUM_THREADS, 1, 1), args=[
			grad_input.numel(), grad_output.data_ptr(), self.argmax.data_ptr(), len(rois), cupy.float32(self.spatial_scale), self.input_size[-3],
			self.input_size[-2], self.input_size[-1], self.pooled_height, self.pooled_width, grad_input.data_ptr(), rois.data_ptr()
			  ], stream=RoiPooling.Stream(ptr=torch.cuda.current_stream().cuda_stream))
		return grad_input, None
		
def overlap(box1, box2):
    b1, b2 = box1.t().contiguous(), box2.t().contiguous()
    xx1 = torch.max(b1[0].unsqueeze(1), b2[0].unsqueeze(0))
    yy1 = torch.max(b1[1].unsqueeze(1), b2[1].unsqueeze(0))
    xx2 = torch.min(b1[2].unsqueeze(1), b2[2].unsqueeze(0))
    yy2 = torch.min(b1[3].unsqueeze(1), b2[3].unsqueeze(0))
    inter = area(x1 = xx1, y1 = yy1, x2 = xx2, y2 = yy2)
    return inter / (area(b1.t()).unsqueeze(1) + area(b2.t()).unsqueeze(0) - inter)

def area(boxes = None, x1 = None, y1 = None, x2 = None, y2 = None):
    return (boxes[..., 3] - boxes[..., 1] + 1) * (boxes[..., 2] - boxes[..., 0] + 1) if boxes is not None else (x2 - x1 + 1).clamp(min = 0) * (y2 - y1 + 1).clamp(min = 0)
```
**Note:** I've also had to replace `utils/bbox.pyx` by `utils/cython_bbox.pyx` and `utils/nms.pyx` by `utils/cython_nms.pyx` in `lib/setup.py` to deal with some `setup.py` issues.
