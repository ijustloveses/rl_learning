环境配置
==========

为了让 gym 工作
```
$ sudo apt install python-tk
$ pip install gym
$ sudo apt-get install python-opengl
```

安装 pytorch
```
$ pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
$ pip install torchvision
```

为了在没有 x 的服务器上运行图形程序，需要使用 xvfb 启动一个虚拟 buffer，否则 gym 的 pyglet.gl 会无法找到窗口，进而报错
```
$ sudo apt install xvfb
$ xvfb-run -s "-screen 0 1400x900x24" python dqn.py
```
然而，并不工作

尝试在 jupyter notebook 上运行程序，仍然不工作
```
$ nohup xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip=0.0.0.0 > nb.log 2>&1
```

经研究，发现是因为 nvidia cuda 安装时驱动和 xvfb 冲突，如下:

- Using xvfb as X-server somehow clashes with the Nvidia drivers. But finally this post pointed me into the right direction. Xvfb works without any problems if you install the Nvidia driver with the -no-opengl-files option and CUDA with --no-opengl-libs option. 

那么，解决方案就是重装 nvidia driver，参考
```
https://gist.github.com/8enmann/931ec2a9dc45fde871d2139a7d1f2d78
```

上面参考文档中，重装了 nvidia driver 和 cuda-8，而实际上我只重装了 nvidia driver，就 OK 了
```
$ sudo chmod +x NVIDIA
$ sudo chmod +x NVIDIA-Linux-x86_64-384.111.run
$ sudo apt-get --purge remove nvidia-*
$ sudo nvidia-uninstall   # 这个实际上没有工作，这是因为之前没有单独安装 nvidia driver，而是装 cuda-8 时自动安装的，故此没有这个卸载文件
$ sudo shutdown -r now
$ sudo service lightdm stop
$ sudo ./NVIDIA-Linux-x86_64-384.111.run --no-opengl-files
```

继续尝试运行
```
$ xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python dqn.py 
```
不报错了，运行完了，但是没有在 Xming 中显示出来

```
$ xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- jupyter notebook --ip=0.0.0.0
```
在 jupyter notebook 中运行成功


catepole/dqn.py
==================
实现了 DQN 算法玩 catepole-v0，而且是标准的 DQN 算法

代码来自 pytorch 的 DQN tutorial

模型采用 CNN，使用本帧和前一帧屏幕 RGB 值之差，而没有使用 gym 提供的物理状态


catepole/a2c.py
=================
Advantage Actor Critic 算法玩 catepole-v0

A2C 就是 A3C 的非异步版本

代码参考文章 [RL实战：用PyTorch 150行代码实现Advantage Actor-Critic玩CartPole](https://zhuanlan.zhihu.com/p/27860621)

actor 和 critic 模型都采用 3 层全连接神经网络，使用 gym 提供的物理状态作为 State
