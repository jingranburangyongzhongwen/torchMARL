# torchMARL
**[English](README.md) | 简体中文**

主要是一些MARL算法的pytorch实现，目前包括：[VDN](https://arxiv.org/abs/1706.05296)、[QMIX](https://arxiv.org/abs/1803.11485)、[QTRAN](https://arxiv.org/abs/1905.05408)、[Qatten](https://arxiv.org/abs/2002.03939)、[Weighted QMIX(CW-QMIX, OW-QMIX)](https://arxiv.org/abs/2006.10800)、[QPLEX](https://arxiv.org/abs/2008.01062)。

该项目基于目前已有的一些实现（[Pymarl](https://github.com/oxwhirl/pymarl), [StarCraft](https://github.com/starry-sky6688/StarCraft), [QPLEX](https://github.com/wjh720/QPLEX)）改进得到，简化了模块与算法流程，改进可视化，方便建立自己的算法库。

网络参数设置与“The StarCraft Multi-Agent Challenge”（[SMAC](https://arxiv.org/abs/1902.04043)）中保持一致。

目前在StarCraft II上进行测试，但可以方便地迁移到任意封装好的环境使用。

`./envs`包含一些QPLEX使用的环境和额外的多目标环境——go_orderly。go_orderly每次都会随机产生一组目标，agents需要到达各自对应的目标才能拿到奖赏，使用这个环境验证一种类似[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)的改进。


## Installation

- Python >= 3.6
- Pytorch >= 1.2
- SMAC
- Seaborn >= 0.9

对于SMAC，这里简单介绍一下linux下的安装，Windows等系统可以查看[他们的仓库](https://github.com/oxwhirl/smac)。

1. 通过下列命令安装SMAC

   `pip install git+https://github.com/oxwhirl/smac.git`

2. 安装StarCraft II，这里给出 [4.6.2](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip) 的下载链接，因为SMAC用的就是这个，并且他说不同版本之间不能比较，其余版本可以查看[暴雪的仓库](https://github.com/Blizzard/s2client-proto)，解压时需要密码`iagreetotheeula`。解压后文件默认路径为`~/StarCraftII/`，如果放在别的路径，需要更改环境变量`SC2PATH`

3. 下载[SMAC MAPS](https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip)，解压后将文件夹直接放在`$SC2PATH/Maps`下即可

4. 运行`python -m smac.bin.map_list `测试地图是否放置成功，运行`python -m smac.examples.random_agents`测试安装是否成功。如果是centos的话可能会出现因为缺少对应版本依赖`/home/user/SMAC/StarCraftII/Versions/Base75689/SC2_x64: /lib64/libc.so.6: version 'GLIBC_2.18' not found (required by /home/user/SMAC/StarCraftII/Libs/libstdc++.so.6)`而导致`pysc2.lib.remote_controller.ConnectError: Failed to connect to the SC2 websocket. Is it up?`，这时候就要根据情况安装依赖或者使用docker了。

## Usage

可以使用以下命令在3s5z_vs_3s6z上运行QMIX实验：

```shell
$ python -u main.py --map='3s5z_vs_3s6z' --alg='qmix' --max_steps=2000000 --epsilon_anneal_steps=50000 --num=5 --gpu='0'
```

或者直接pycharm打开项目，run main.py即可。也可以使用run.sh复现QMIX与HER改进在go_orderly上的实验。

SMAC的各种地图描述在这里：https://github.com/oxwhirl/smac/blob/master/docs/smac.md

## Results

所有地图的环境设置均与SMAC相同，难度为7（VeryHard）

`./imgs` 会有一些额外的图片，是之前版本代码的结果。

### 3s_vs_5z QMIX

<img src="./imgs/qmix-3s_vs_5z.png" style="zoom:80%;" />

### go_orderly

| <img src="./imgs/rewards.png" style="zoom:80%;" /> | <img src="./imgs/win_rates.png" style="zoom:80%;" /> |
| :------------------------------------------------: | :-------------------------------------------------: |
|            累加奖赏           |            胜率             |

## Replay

If you want to see the replay, make sure the `replay_dir` is an absolute path, which can be set in `./common/arguments.py`. Then the replays of each evaluation will be saved, you can find them in your path.