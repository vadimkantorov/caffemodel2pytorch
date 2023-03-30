from functools import reduce


class Blob(object):
    AssignmentAdapter = type(
        "",
        (object,),
        dict(
            shape=property(lambda self: self.contents.shape),
            __setitem__=lambda self, indices, values: setattr(self, "contents", values),
        ),
    )

    def __init__(self, data=None, diff=None, numpy=False):
        self.data_ = data if data is not None else Blob.AssignmentAdapter()
        self.diff_ = diff if diff is not None else Blob.AssignmentAdapter()
        self.shape_ = None
        self.numpy = numpy

    def reshape(self, *args):
        self.shape_ = args

    def count(self, *axis):
        return reduce(lambda x, y: x * y, self.shape_[slice(*(axis + [-1])[:2])])

    @property
    def data(self):
        if self.numpy and torch.is_tensor(self.data_):
            self.data_ = self.data_.detach().cpu().numpy()
        return self.data_

    @property
    def diff(self):
        if self.numpy and torch.is_tensor(self.diff_):
            self.diff_ = self.diff_.detach().cpu().numpy()
        return self.diff_

    @property
    def shape(self):
        return self.shape_ if self.shape_ is not None else self.data_.shape

    @property
    def num(self):
        return self.shape[0]

    @property
    def channels(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[2]

    @property
    def width(self):
        return self.shape[3]
