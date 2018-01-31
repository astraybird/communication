## mxnet module.py类

```module.py```是一个重要的类，一个完整的训练过程都要经过这个类

### 构建模型

```
>>> data = mx.sym.Variable('data')
>>> fc1  = mx.sym.FullyConnected(data, name='fc1', num_hidden=128)
>>> act1 = mx.sym.Activation(fc1, name='relu1', act_type="relu")
>>> fc2  = mx.sym.FullyConnected(act1, name='fc2', num_hidden=10)
>>> out  = mx.sym.SoftmaxOutput(fc2, name = 'softmax')
>>> mod = mx.mod.Module(out)
>>> out
<Symbol softmax>
>>> mod
<mxnet.module.module.Module object at 0x7f52df417c50>
>>>
```

上述使用符号式编程构建了一个计算图，SoftmaxOutput带了loss

### bind

假设这里已经加载出imagenet的数据集data_set, 包含两个MXDataIter的数据，一个train,一个val,

```
>>> mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
```

```
>>> data_shapes
[DataDesc[data,(128L, 3L, 224L, 224L),<type 'numpy.float32'>,NCHW]]
>>> mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
>>> label_shapes = train.provide_label
>>> label_shapes
[DataDesc[softmax_label,(128L,),<type 'numpy.float32'>,NCHW]]
```

把初始化module时候的symbols绑定到construct executors,并初始化内存

bind首先得到DataDesc类型的_data_shape,\_label_shape(通过参数传递进来),\_symbol,\_context(初始化的时候得到),这些参数传给DataParallelExecutorGroup, 得到\_exec_group对象，这是个重要的对象，如果参数已经初始化，就将初始化后参数和辅助参数设置给\_exec_group对象

下面整理一下executor_group.py这个类及其所对应的对象DataParallelExecutorGroup

接收参数：

- symbol,构建的模型计算图
- context,设备，mx.cpu(),mx.gpu(0),mx.gpu(1)...
- data_shapes,DataDesc类型，包括了数据的形状
- label_shape,同上
- param_names,详情见下方的解释
- 其他参数...

全局变量：

- param_names,参数的名称，用户定义模型的时候如果写了name，系统会自动在name后加上weight,bias,gamma,beta等参数，可以理解为卷积核的名称

```
 000 = {str} 'bn_data_gamma'
 001 = {str} 'bn_data_beta'
 002 = {str} 'conv0_weight'
 003 = {str} 'bn0_gamma'
 004 = {str} 'bn0_beta'
 005 = {str} 'stage1_unit1_bn1_gamma'
 006 = {str} 'stage1_unit1_bn1_beta'
 007 = {str} 'stage1_unit1_conv1_weight'
 008 = {str} 'stage1_unit1_bn2_gamma'
 009 = {str} 'stage1_unit1_bn2_beta'
 010 = {str} 'stage1_unit1_conv2_weight'
 011 = {str} 'stage1_unit1_bn3_gamma'
 .
 .
 .
 151 = {str} 'stage4_unit3_bn3_beta'
 152 = {str} 'stage4_unit3_conv3_weight'
 153 = {str} 'bn1_gamma'
 154 = {str} 'bn1_beta'
 155 = {str} 'fc1_weight'
 156 = {str} 'fc1_bias'
```



- arg_names-从symbol拿出所有的参数的名称，这个和param_names应该是一模一样的，不知道为什么会重复来一个
- aux_names

```
self.aux_names = {list} <type 'list'>: ['bn_data_moving_mean', 'bn_data_moving_var', 'bn0_moving_mean', 'bn0_moving_var', 'stage1_unit1_bn1_moving_mean', 'stage1_unit1_bn1_moving_var', 'stage1_unit1_bn2_moving_mean', 'stage1_unit1_bn2_moving_var', 'stage1_unit1_bn3_moving_mean', 
 __len__ = {int} 102
 000 = {str} 'bn_data_moving_mean'
 001 = {str} 'bn_data_moving_var'
 002 = {str} 'bn0_moving_mean'
 003 = {str} 'bn0_moving_var'
 004 = {str} 'stage1_unit1_bn1_moving_mean'
 005 = {str} 'stage1_unit1_bn1_moving_var'
 006 = {str} 'stage1_unit1_bn2_moving_mean'
 007 = {str} 'stage1_unit1_bn2_moving_var'
 098 = {str} 'stage4_unit3_bn3_moving_mean'
 099 = {str} 'stage4_unit3_bn3_moving_var'
 100 = {str} 'bn1_moving_mean'
 101 = {str} 'bn1_moving_var'
```

- symbol,计算图
- workload,不明白是个什么，可能是设备相关吧，打印出来

```
self.workload = {list} <type 'list'>: [1, 1, 1, 1]
 __len__ = {int} 4
 0 = {int} 1
 1 = {int} 1
 2 = {int} 1
 3 = {int} 1
```

- for_training,是否训练
- 输入数据是否需要梯度
- _total_exec_bytes每个设备所占的显存，一个工具变量，profiler会用到
- fixed_param_names,指定哪些操作的参数固定住不动
- slices,数据被分成多少分，最外层所有的batch_size被分成GPU个数的等分，在batch_size的维度划分
- execs, Executor对象，有多少个设备就有多少个该对象
- _default_execs,默认的executor对象，默认是空
- data_array,输入的数据，但是这里全是0，应该是一个占位符，根据shape初始化内存用的
- label_arrays,同上
- param_arrays,参数被全部初始化为0，这些参数在executor_group.py的bind_exec执行完后得以初始化
- state_arrays,不知道是啥
- grad_arrays，形状与param_arrays一样，grad_arrays更新param_arrays
- aux_arrays
- input_grad_arrays
- self.data_shapes,输入数据的大小
- label_shapes,同上
- data_names,输入数据的名称，一般是'data'
- label_names,输入label的名称，一般是'softmax_label'
- data_layouts
- label_layouts
- output_names
- output_layouts
- num_outputs, resnet50是一个，应该是输出，可能是最终的输出，目前不知道



execs-mxnet.executor.Executor对象，有几个设备就有几个对象，比如四个GPU，就四个这样对象的列表

grad_arrays-也绑定在各个设备上，目测是用于更新参数的梯度，每个GPU计算自己的梯度

param_names-参数列表名称，比如每个卷积的卷积核weight, bias

有计算图symbol,data_shape,label_shape，那么整个网络的细节就已经清楚

- _bind_ith_exec方法

grad_requirement, 'write', 'add'或'null',add是把所有的梯度加起来，wirte是什么操作？

symbol.py的check_call(_LIB.MXExecutorSimpleBind())这个函数干了什么？

该方法内部根据设备的个数得到executor，初始化一个executor会初始化参数和辅助参数

执行完bind_exec的\_collect_arrays，executor_group的data_arrays得到初始化

## init_params

执行完init_params函数，module对象的\_exec_group变量的param_arrays对象就不再全部是0，而是用户传递进来的， init_params直接调用\_exec_group对象的set_params方法，然后对每个executor执行初始化

## init_optimizer

module的init_optimizer方法根据用户传递进来的optimier的字符串和参数，通过optimizer.py创建一个optimizer对象

此外该方法初始化了kvstore,```_initialize_kvstore```，这个还没有弄明白

## forward

module.py->forward()->executor_group.py->forward->为每一个设备的Executor执行forward->executor.py->forward,这里调用C++的MXExecutorForward，执行后，executor.py的self.outputs得到输出，每个输出的大小是32x1000，每一张图片一次前向计算得到1000个概率值，代表该次计算是某一个类的概率

## backword

module.py->backward->executor_group.py->backward()->executor.py->MXExecutorBackwardEx(),执行反向计算，反向计算得到梯度self.grad_arrays,存储在每一个executor对象

## update

通过optimizer更新梯度到参数，梯度是之前前向-后向计算得到的张量，DataParallelExecutorGroup对象持有设备数量个Executor,每个executor对象持有一个分grad_arrays,这个地方有些疑惑，update方法传递进去的是\_exec_group.grad_arrays, 将每一个executor上的梯度传递给\_exec_group.grad_arrays的方法\_collect_arrays只有bind_exec的时候执行过一次，backward方法完了并没有看到\_exec_group去收集各个executor的梯度。--这个问题的可能解释是bind执行的那次把每一个executor的梯度给 exec_group的梯度是指针或者引用，那每次更新了每个executor的梯度，exec_group的梯度值自然就得到了更新

最终梯度聚合在kvstore_dist.h的Push_方法里将梯度相加

和C++怎么交互的还没看明白，看明白的大神请给我留个文章链接，我来研究研究

参考http://www.cnblogs.com/heguanyou/p/7604326.html

