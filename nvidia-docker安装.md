## nvidia-docker安装

#### 环境

ubuntu18.04

#### 配置华为证书

```
sudo cp Huawei* /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

#### 安装docker-ce

如果版本低于18会比较麻烦，要17以上的

```
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get install docker-ce
```

#### 安装nvidia-docker

```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

sudo apt-get install nvidia-docker2
```
####配置代理
```
mkdir -p /etc/systemd/system/docker.service.d
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:80/"
sudo systemctl daemon-reload
sudo systemctl restart docker
```



参考：

基于官方docker

```
https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository
http://3ms.huawei.com/km/blogs/details/5485603
```



