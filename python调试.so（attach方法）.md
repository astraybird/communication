## python调试.so（attach方法）

### 编译

修改config.mk

```ADD_CFLAGS = -g -O0```

### 启动python进程

```gdb --args python train_cifar10.py --network=mlp --gpus=0,1,2,3```

设置断点：

```B mxnet::kvstore::KVStoreLocal::Push```

```r```继续执行直到停止

