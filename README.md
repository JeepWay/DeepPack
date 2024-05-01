# Deep-Pack in Pytorch

This repository contains a Pytorch **unofficial** implementation of the algorithm presented in the paper [Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8956393)


## Modified CNN Architecture
Because the dimension of the proposed concatenated state (W×2H) inputted to the model doesn't satisfyied the dimension requirements of the CNN architecture mentioned in the paper, thus, this repository use the modified CNN architecture to train. 

:arrow_down: CNN architecture for 5*5 bin size
<p align="left">
<img src="img/CNN_5x5.jpg" alt="CNN for 5*5 bin size" height="40%" width="70%">
</p>

:arrow_down: CNN architecture for 4*4 bin size
<p align="left">
<img src="img/CNN_4x4.jpg" alt="CNN for 5*5 bin size" height="40%" width="70%">
</p>


## Dataset 
The all training and testing dataset are generated from `generate_train.py` and `generate_test.py` with specific random seed, and `data` folder contain only training and testing data for 5x5 bin size, to reduce the size of the `data` folder.

If you want to see the training and testing data for 4x4 or 3x3 bin size, you can run `generate_train.py` and `generate_test.py` to generate it.


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
* if you want to use your own env, then run:
    ```
    pip install -r install/requirements.txt
    ```
* if you want to create new env, then run:
    ```
    conda env create -f install/environment.yml
    ```
    or run: 
    ```
    .\install\install.bat
    ```


## Usage example
* `run.bat`
    * Execute the command in the `.bat` file, including training and testing part.
* `python main.py` 
    * Execute to train the doubleDQN model.
* `python test.py` 
    * Execute to test the result of the trained doubleDQN model.



## Results Folder
The all training and testing result and log will be placed in `(items_name)_(bin_w)_(bin_h)_result` folder.

```bash
├── (itemsName)_(binW)_(binH)_result
|   ├── img
|   |   ├── train_record.png
|   ├── test
|   |   ├── log.txt
|   ├── train
|   |   ├── log.txt
```

## Experimental Results
### Train
We only show the result of 4x4 and 5x5 bin size, just like the original paper. If you want to see the result of 3x3 bin size, you can generate 3x3 training data and then train it with suitable hyper-parameters.

We plot the training curves with moving average (window size 50).

:arrow_down: training result of 4*4 bin size
| loss                                                                                                  | reward                                                                                                   | PE                                                                                                  |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| ![unit_square_4x4](unit_square_4x4_result/img/iterations_training_loss.png) | ![unit_square_4x4](unit_square_4x4_result/img/iterations_training_rewards.png) | ![unit_square_4x4](unit_square_4x4_result/img/iterations_training_PE.png) |
| ![square_4x4](square_4x4_result/img/iterations_training_loss.png)           | ![square_4x4](square_4x4_result/img/iterations_training_rewards.png)           | ![square_4x4](square_4x4_result/img/iterations_training_PE.png)           |
| ![rectangular_4x4](rectangular_4x4_result/img/iterations_training_loss.png) | ![rectangular_4x4](rectangular_4x4_result/img/iterations_training_rewards.png) | ![rectangular_4x4](rectangular_4x4_result/img/iterations_training_PE.png) |


:arrow_down: training result of 5*5 bin size
| loss | reward | PE  |
| ---- | ------ | --- |
|      |        |     |












### Test
In test section, we only show the result of 4x4 and 5x5 bin size, just like the original paper.

Because the training data which are randomly generated may not contain the specific test sequence types, leading to the model doesn't see that pattern, therefore, the result of the test may not perform as good as the training do.

We guess the the training data in the original paper contain the specific test sequence types, thus, the test result of the the original paper is awesome, not like our experiments.





## References
* [Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8956393)
