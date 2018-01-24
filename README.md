## mxnet加载数据的方式

可用于数据训练的数据格式：

- .rec文件
- raw image, 
- .lst文件，用mx.image.ImageIter接口读取，这个接口也可读.rec的文件

大概过程：

传data iter对象给base_module，把数据对象变为iterator, (对象iterable不一定是iterator), ```data_iter = iter(train_data)```

### 使用mx.io读取数据

1. 从内存中读取

调用```mx.io.NDArrayIter```接口

mxnet官方例子的train_mnist.py

```
def read_data(label, image):
   """
   download and read data into numpy
   """
   base_url = 'http://yann.lecun.com/exdb/mnist/'
   with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
       magic, num = struct.unpack(">II", flbl.read(8))
       label = np.fromstring(flbl.read(), dtype=np.int8)
   with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
       magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
       image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
   return (label, image)


def to4d(img):
   """
   reshape to 4D arrays
   """
   return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
   """
   create data iterator with NDArrayIter
   """
   (train_lbl, train_img) = read_data(
           'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
   (val_lbl, val_img) = read_data(
           't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
   train = mx.io.NDArrayIter(
       to4d(train_img), train_lbl, args.batch_size, shuffle=True)
   val = mx.io.NDArrayIter(
       to4d(val_img), val_lbl, args.batch_size)
   return (train, val)
#参考train_mnist.py
```

2. 从CSV文件中读取数据

```
#lets save `data` into a csv file first and try reading it back
np.savetxt('data.csv', data, delimiter=',')
data_iter = mx.io.CSVIter(data_csv='data.csv', data_shape=(3,), batch_size=30)
for batch in data_iter:
   print([batch.data, batch.pad])
```

3. 使用mx.io.ImageRecordIter接口,需要制定kvstore

```
   train = mx.io.ImageRecordIter(
       path_imgrec         = args.data_train,
       path_imgidx         = args.data_train_idx,
       label_width         = 1,
       mean_r              = rgb_mean[0],
       mean_g              = rgb_mean[1],
       mean_b              = rgb_mean[2],
       data_name           = 'data',
       label_name          = 'softmax_label',
       data_shape          = image_shape,
       batch_size          = args.batch_size,
       rand_crop           = args.random_crop,
       max_random_scale    = args.max_random_scale,
       pad                 = args.pad_size,
       fill_value          = 127,
       min_random_scale    = args.min_random_scale,
       max_aspect_ratio    = args.max_random_aspect_ratio,
       random_h            = args.max_random_h,
       random_s            = args.max_random_s,
       random_l            = args.max_random_l,
       max_rotate_angle    = args.max_random_rotate_angle,
       max_shear_ratio     = args.max_random_shear_ratio,
       rand_mirror         = args.random_mirror,
       preprocess_threads  = args.data_nthreads,
       shuffle             = True,
       num_parts           = nworker,
       part_index          = rank)
```

对于gluon使用数据的方式更灵活约束更少但是也更麻烦，从```ImageRecordIter```获取的是一个Iterator的对象，在gluon里面这样获取数据：

```
for i, batch in enumerate(train_data):
       data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
       label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
```

每次迭代出来的是```mx.io.DataBatch```对象:

```
>>> for i, batch in enumerate(train):
...     print batch
...     print batch.data[0].shape
...     break
...
DataBatch: data shapes: [(128L, 3L, 224L, 224L)] label shapes: [(128L,)]
(128L, 3L, 224L, 224L)
>>>
```


在```base_module.py```中通过next一次获取一个batch数据：

```
>>> train_data
<mxnet.io.MXDataIter object at 0x7f52df4171d0>
>>> data_iter = iter(train_data)
>>> next_data_batch = next(data_iter)
>>> next_data_batch
<mxnet.io.DataBatch object at 0x7f52df417450>
>>> data = next_data_batch.data[0]
>>> data.shape
(128L, 3L, 224L, 224L)
>>>
```

mx.io.DataBatch中获取的data为什么要通过下标[0]去获取，我也没看明白，只有0号下标，没有1或者更多

### 使用mx.image读取数据

自己用的比较少，从imagelist中读取可迭代的数据：

```
data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 224, 224), label_width=1,
data_iter.reset()
for data in data_iter:
d = data.data[0]
print(d.shape)
# we can apply lots of augmentations as well
data_iter = mx.image.ImageIter(4, (3, 224, 224), path_imglist='data/custom.lst',
data = data_iter.next()
# specify augmenters manually is also supported
data_iter = mx.image.ImageIter(32, (3, 224, 224), path_rec='data/caltech.rec',
```

### 使用gluon接口读取数据

使用example/gluon下面举例：

- 构造gluon.data.Dataset：

```
class CIFAR10(_DownloadedDataset):
  """CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html

  Each sample is an image (in 3D NDArray) with shape (32, 32, 1).

  Parameters
  ----------
  root : str, default '~/.mxnet/datasets/cifar10'
      Path to temp folder for storing data.
  train : bool, default True
      Whether to load the training or testing set.
  transform : function, default None
      A user defined callback that transforms each sample. For example:
  ::

      transform=lambda data, label: (data.astype(np.float32)/255, label)

  """
  def __init__(self, root='~/.mxnet/datasets/cifar10', train=True,
               transform=None):
      self._archive_file = ('cifar-10-binary.tar.gz', 'fab780a1e191a7eda0f345501ccd62d20f7ed891')
      self._train_data = [('data_batch_1.bin', 'aadd24acce27caa71bf4b10992e9e7b2d74c2540'),
                          ('data_batch_2.bin', 'c0ba65cce70568cd57b4e03e9ac8d2a5367c1795'),
                          ('data_batch_3.bin', '1dd00a74ab1d17a6e7d73e185b69dbf31242f295'),
                          ('data_batch_4.bin', 'aab85764eb3584312d3c7f65fd2fd016e36a258e'),
                          ('data_batch_5.bin', '26e2849e66a845b7f1e4614ae70f4889ae604628')]
      self._test_data = [('test_batch.bin', '67eb016db431130d61cd03c7ad570b013799c88c')]
      super(CIFAR10, self).__init__('cifar10', root, train, transform)

  def _read_batch(self, filename):
      with open(filename, 'rb') as fin:
          data = np.fromstring(fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

      return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
             data[:, 0].astype(np.int32)

  def _get_data(self):
      if any(not os.path.exists(path) or not check_sha1(path, sha1)
             for path, sha1 in ((os.path.join(self._root, name), sha1)
                                for name, sha1 in self._train_data + self._test_data)):
          filename = download(self._get_url(self._archive_file[0]),
                              path=self._root,
                              sha1_hash=self._archive_file[1])

          with tarfile.open(filename) as tar:
              tar.extractall(self._root)

      if self._train:
          data_files = self._train_data
      else:
          data_files = self._test_data
      data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                          for name, _ in data_files))
      data = np.concatenate(data)
      label = np.concatenate(label)

      self._data = nd.array(data, dtype=data.dtype)
      self._label = label
```

构造gluon.data.DataLoader:

```
train_data = gluon.data.DataLoader(
  gluon.data.vision.MNIST('./data', train=True, transform=transformer),
  batch_size=opt.batch_size, shuffle=True, last_batch='discard')
```

训练时迭代出数据：

```
for i, batch in enumerate(train_data):
  //todo
```

  ​
