# DRL-OR: Deep Reinforcement Learning-based Online Routing for Multi-type Service Requirements

This is a Pytorch implementation of DRL-OR on INFOCOM 2021. 

## Requirements

To run DRL-OR

* Firstly you should install [mininet](http://mininet.org/download/)

```
git clone git://github.com/mininet/mininet
cd mininet
util/install.sh -a
```

* Then install ryu-manager and other required packages

```
pip3 install ryu
cd drl-or
pip3 install requirements.txt
```

## Running DRL-OR

To run DRL-OR code as an example

* Run testbed

```
cd testbed
sudo ./run.sh
```

* Run ryu controller

```
cd ryu-controller
./run.sh
```

* Run DRL-OR algorithm

```
cd drl-or
./run.sh
```

If you have any questions, please post an issue or send an email to chenyiliu9@gmail.com.

## Citation

```
@inproceedings{liu2021drl-or,
  title={DRL-OR: Deep Reinforcement Learning-based Online Routing for Multi-type Service Requirements},
  author={Liu, Chenyi and Xu, Mingwei and Yang, Yuan and Geng, Nan},
  booktitle={Proc. IEEE INFOCOM},
  year={2021}
}
```



