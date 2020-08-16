# torchMARL

主要是一些MARL算法的pytorch实现，目前包括：
[VDN](https://arxiv.org/abs/1706.05296), [QMIX](https://arxiv.org/abs/1803.11485), [weighted QMIX(CWQMIX, OWQMIX)](https://arxiv.org/abs/2006.10800)

该项目基于 https://github.com/starry-sky6688/StarCraft 改进得到，简化了模块与算法流程，改进可视化，方便建立自己的算法库。

目前在SMAC上进行测试，可以方便地迁移到任意封装好的环境使用。

## Corresponding Papers

- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)
- [From Few to More: Large-scale Dynamic Multiagent Curriculum Learning](https://arxiv.org/abs/1909.02790?context=cs.MA)
- [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://arxiv.org/abs/1911.10715)
- [MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/abs/1910.07483)
- [Rashid, Tabish, et al. “Weighted QMIX: Expanding Monotonic Value Function Factorisation.” ArXiv Preprint ArXiv:2006.10800, 2020.](https://arxiv.org/abs/2006.10800)

## Requirements

- python
- Pytorch
- [SMAC](https://github.com/oxwhirl/smac)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)
+ [pymarl](https://github.com/oxwhirl/pymarl)


## TODO List

- [ ] 调整Weighted QMIX，目前无法复现其5m_vs_6m实验结果
- [ ] Qatten
- [ ] Other SOTA MARL algorithms
- [ ] Update results on other maps

## Quick Start

```shell
$ python main.py --map=3m --alg=qmix
```

或者直接pycharm打开项目，run main.py即可，默认参数是qmix进行3m场景的训练。

SMAC的各种地图描述在这里：https://github.com/oxwhirl/smac/blob/master/docs/smac.md

## Result

暂时只贴一部分，因为我目前主要实现值分解的算法，还在实现新的。

### 1. QMIX 3m --difficulty=7(VeryHard)
![qmix-3m-7](./img/qmix-3m-7.png)

### 2. VDN 3m --difficulty=7(VeryHard)

![vdn-3m-7](./img/vdn-3m-7.png)

## Replay

If you want to see the replay, make sure the `replay_dir` is an absolute path, which can be set in `./common/arguments.py`. Then the replays of each evaluation will be saved, you can find them in your path.
