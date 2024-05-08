# Deep-Pack in PyTorch
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/JeepWay/DeepPack/blob/main/LICENSE)  

This repository contains a Pytorch **unofficial** implementation of the algorithm presented in the paper [Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8956393)


## Modified CNN Architecture
Because the dimension of the proposed concatenated state (W×2H) inputted to the model doesn't satisfyied the dimension requirements of the CNN architecture mentioned in the paper, thus, this repository use the modified CNN architecture to train. 

:arrow_down: CNN architecture for 4x4 bin size
<p align="left">
<img src="img/CNN_4x4.jpg" alt="CNN for 4x4 bin size" height="40%" width="100%">
</p>

:arrow_down: CNN architecture for 5x5 bin size
<p align="left">
<img src="img/CNN_5x5.jpg" alt="CNN for 5x5 bin size" height="40%" width="100%">
</p>


## Dataset 
The all training and testing dataset are generated from `generate_train.py` with random seed 777 and `generate_test.py` with random seed 666.

To reduce the size of this repository, we don't upload the training and testing data except for 4x4 bin size with retangular item.

If you want to see the training and testing data, you can run `generate_train.py` and `generate_test.py` to generate it.


## Task type
* Bin size
    * 3x3
    * 4x4
    * 5x5
* Items type
    * Unit square (1x1)
    * Only square
    * Rectangular and square


## Environments
* OS: Window 10
* Visual Studio Code
* Anaconda3
* Python: 3.9.15
* PyTorch: 2.X.X+cu118


## Installation
* If you want to use your own env, then run:
    ```
    pip install -r install/requirements.txt
    ```
* If you want to create new env, then run:
    ```
    conda env create -f install/environment.yml
    ```
    or run: 
    ```
    .\install\install.bat
    ```


## Usage example
* `run.bat`
    * Execute the command in the `.bat` file, including training and testing parts.
* `python main.py --task rectangular --bin_w 4 --bin_h 4 --iterations 300000` 
    * Execute to train the doubleDQN model.
* `python test.py --task rectangular --bin_w 4 --bin_h 4 --sequence_type type1 --max_sequence 4  --iterations 5000` 
    * Execute to test the result of the trained doubleDQN model.


## Result Folder
The all training and testing result and log will be placed in `(items_name)_(bin_w)x(bin_h)_result` folder. 

To reduce the size of this repository, we don't upload the result folders.

If you want to see the traing log and testing result and dynamic demo gifs, you can run `.\install\install.bat` or other command lines, describes in `Installation` section, then you will get the whole information.

```bash
├── (itemsName)_(binW)x(binH)_result
|   ├── img
|   |   ├── train_record.png
|   |   ├── test_record.png
|   |   ├── test_type*.gif
|   ├── test
|   |   ├── log.txt
|   |   ├── test_result.txt
|   ├── train
|   |   ├── log.txt
|   ├── hyperparameter.txt
|   ├── (itemsName)_(binW)x(binH).pt
```


## Experimental Results
### Train
We only show the result of 4x4 and 5x5 bin size, just like the original paper. If you want to see the result of 3x3 bin size, you can generate 3x3 training data and then train it with suitable hyper-parameters.

We plot the training curves with moving average (window size 50).

#### Training result of 4x4 bin size
| loss                                                                            | reward                                                                             | PE                                                                            |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| ![unit_square_4x4](img/unit_square_4x4_result/img/iterations_training_loss.png) | ![unit_square_4x4](img/unit_square_4x4_result/img/iterations_training_rewards.png) | ![unit_square_4x4](img/unit_square_4x4_result/img/iterations_training_PE.png) |
| ![square_4x4](img/square_4x4_result/img/iterations_training_loss.png)           | ![square_4x4](img/square_4x4_result/img/iterations_training_rewards.png)           | ![square_4x4](img/square_4x4_result/img/iterations_training_PE.png)           |
| ![rectangular_4x4](img/rectangular_4x4_result/img/iterations_training_loss.png) | ![rectangular_4x4](img/rectangular_4x4_result/img/iterations_training_rewards.png) | ![rectangular_4x4](img/rectangular_4x4_result/img/iterations_training_PE.png) |


#### Training result of 5x5 bin size
| loss                                                                            | reward                                                                             | PE                                                                            |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| ![unit_square_5x5](img/unit_square_5x5_result/img/iterations_training_loss.png) | ![unit_square_5x5](img/unit_square_5x5_result/img/iterations_training_rewards.png) | ![unit_square_5x5](img/unit_square_5x5_result/img/iterations_training_PE.png) |
| ![square_5x5](img/square_5x5_result/img/iterations_training_loss.png)           | ![square_5x5](img/square_5x5_result/img/iterations_training_rewards.png)           | ![square_5x5](img/square_5x5_result/img/iterations_training_PE.png)           |
| ![rectangular_5x5](img/rectangular_5x5_result/img/iterations_training_loss.png) | ![rectangular_5x5](img/rectangular_5x5_result/img/iterations_training_rewards.png) | ![rectangular_5x5](img/rectangular_5x5_result/img/iterations_training_PE.png) |


### Test
In test section, we only show the result of 4x4 and 5x5 bin size, just like the original paper, but we have modified some wrong sequence types of the original paper.

Because the training data which are randomly generated may not contain the specific test sequence types, leading to the model doesn't see that pattern, therefore, the result of the test may not perform as good as the training do.

We guess the the training data in the original paper contain the specific test sequence types, thus, the test result of the the original paper is awesome, not like our experiments.

<p align="center">
<img src="img/test_result.jpg" alt="test_result" height="40%" width="100%">
</p>


## Dynamic Demo Example
####  Unit square 4x4 bin with type1 sequence 
<p align="center">
<img src="img/unit_square_4x4_result/img/test_type1.gif" alt="unit_square_4x4_type1" height="40%" width="80%">
</p>

####  Square 4x4 bin with type1 sequence 
<p align="center">
<img src="img/square_4x4_result/img/test_type3.gif" alt="square_4x4_type3" height="40%" width="80%">
</p>

####  Rectangular 4x4 bin with type2 sequence 
<p align="center">
<img src="img/rectangular_4x4_result/img/test_type2.gif" alt="rectangular_4x4_type2" height="40%" width="80%">
</p>

####  Unit square 5x5 bin with type 1 sequence 
<p align="center">
<img src="img/unit_square_5x5_result/img/test_type1.gif" alt="unit_square_5x5_type1" height="40%" width="80%">
</p>

####  Square 5x5 bin with type 3 sequence 
<p align="center">
<img src="img/square_5x5_result/img/test_type3.gif" alt="square_5x5_type3" height="40%" width="80%">
</p>

####  Rectangular 5x5 bin with type 3 sequence 
<p align="center">
<img src="img/rectangular_5x5_result/img/test_type3.gif" alt="rectangular_5x5_type3" height="40%" width="80%">
</p>


## References
* [Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8956393)

* [Double DQN](https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95#83-double-dqn-%E4%BB%A3%E7%A0%81%E5%AE%9E%E8%B7%B5)

* [TD3 github](https://github.com/sfujim/TD3)

* [RainbowDQN github](https://github.com/Kaixhin/Rainbow)

