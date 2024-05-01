import argparse
import time
from matplotlib import pyplot as plt
import torch
from collections import deque
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

from agent import Agent
import utils


def preargparse():
    parser = argparse.ArgumentParser(description='This is a simple program to demonstrate argparse')
    # hyper-parameters mentioned in the paper
    parser.add_argument("--bin_w", default=5, type=int, help="width of bin", choices=[3, 4, 5])
    parser.add_argument("--bin_h", default=5, type=int, help="height of bin", choices=[3, 4, 5])
    parser.add_argument("--task", default="unit_square", type=str, help="task name", choices=["unit_square", "rectangular", "square"])
    parser.add_argument("--learning_rate", default=1e-3, type=int)
    parser.add_argument("--gamma", default=0.95, type=int)
    parser.add_argument("--mem_size", default=20000, type=int, help="replay buffer size")
    parser.add_argument("--K", default=2, type=int, help="coefficient of bonus PE reward for last step")
    parser.add_argument("--epsilon", default=1., type=float)
    parser.add_argument("--iterations", default=1000, type=int, help="each iteration consists of w*h items", choices=[200000, 300000, 350000])
    # other hyper-parameters
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learn_start", default=2000, type=int)
    parser.add_argument("--sync_freq", default=2000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--no_action_reward", default=-5, type=int, choices=[-5, 0])
    args = parser.parse_args()
    args.action_space = args.bin_w * args.bin_h + 1
    args.max_sequence = args.bin_w * args.bin_h
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args



if __name__ == '__main__':
    begin_time = time.asctime(time.localtime(time.time()))
    start = time.time()
    args = preargparse()

    dir_name, date_string = utils.makeTestDir(args.task, args.bin_w, args.bin_h, args.iterations)
    print(date_string, dir_name)
    utils.setSeed(args.seed)
    utils.deleteLastOutput(dir_name)

    replay = deque(maxlen=args.mem_size)
    agent = Agent(args)
    step_losses = []
    iteration_losses = []
    step_rewards = []
    iteration_rewards = []
    PE_list = []
    sequence_count_list = []
    action_list = []

    ''' read train data '''
    file = open(f"./data/train_{args.task}_{str(args.bin_w)}_{args.iterations}.txt",'r')
    content = file.read()
    file.close()
    items = content.split('\n')
    if items[-1] == "":
        items.pop()


    total_steps = 0
    for iter in tqdm(range(args.iterations)):
    # for iter in tqdm(range(100000)):
        sequence_count = 0
        status = 1
        rewards = 0
        bin_state = torch.from_numpy(np.ones((args.bin_w, args.bin_h), dtype=int))

        while(status == 1):
            ''' get item and state of item '''
            new_item = items[iter * args.max_sequence + sequence_count].split()
            width = int(new_item[0])
            height = int(new_item[1])
            item_state = torch.from_numpy(np.ones((args.bin_w, args.bin_h), dtype=int))
            item_state[0:width, 0:height] = 0

            ''' concate bin and item into state image(c=1, h, w)'''
            state = torch.cat((bin_state, item_state), 1).unsqueeze(0)  
        

            ''' get action '''
            if (np.random.random() < args.epsilon):
                action = np.random.randint(0,args.action_space)
            else:
                action = agent.act(state.unsqueeze(0).float().to(args.device))
            action_list.append(action)

            ''' calculate reward then update bin_state'''
            if (action != (args.action_space-1)):
                x = action // args.bin_h
                y = action % args.bin_h
                
                if (x+width > args.bin_w) or (y+height > args.bin_h):   # 判斷是否超出邊界
                    #print(f"Out of boundary: action:{action}, x:{x}, y:{y}, width:{width}, height:{height}")
                    reward = -5
                elif (torch.sum(bin_state[x:x+width, y:y+height]) != (width * height)): # 判斷是否有重疊
                    #print(f"Overlap: action:{action}, x:{x}, y:{y}, width:{width}, height:{height}")
                    reward = -5
                else:       # 合法動作
                    # 更新 bin_state
                    bin_state[x:x+width, y:y+height] = 0
                    # 1 變 0，0 變 1，才能使用 scipy 的 label 函數來找出連通區域
                    state_np = (1 - bin_state).numpy()
                    # 使用 scipy 的 label 函數來找出連通區域
                    # structure 定義了 4-連通性
                    labeled_array, num_features = label(state_np, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
                    # 統計每個連通區域的大小，包含 label 為 0 的部分
                    cluster_sizes = np.bincount(labeled_array.ravel())
                    # 利用 item 放入位置取得連通區域的 label
                    item_label = labeled_array[x][y]
                    cluster_size = cluster_sizes[item_label]
                    # 計算最小包圍矩形的大小（找到群集的極端點）
                    cluster_indices = np.argwhere(labeled_array == item_label)
                    top_left = cluster_indices.min(axis=0)
                    bottom_right = cluster_indices.max(axis=0)
                    bounding_box_size = (bottom_right[0] - top_left[0] + 1) * (bottom_right[1] - top_left[1] + 1)
                    # 計算 compactness
                    compactness = cluster_size / bounding_box_size
                    reward = cluster_size * compactness
            else: # 不放入 item
                reward = -5
            # additional reward for last step
            if (sequence_count+1 == (args.max_sequence)) or (torch.sum(bin_state) == 0):
                PE = torch.sum(1-bin_state) / (args.bin_w * args.bin_h)
                reward += args.K * PE
            step_rewards.append(reward)
            rewards += reward
            

            ''' get next item and state of next item '''
            if (iter * args.max_sequence + sequence_count + 1) == (args.bin_w * args.bin_h * args.iterations):
                new_item = (0,0)
            else:
                new_item = items[iter * args.max_sequence + sequence_count + 1].split()
            width = int(new_item[0])
            height = int(new_item[1])
            item_state2 = torch.from_numpy(np.ones((args.bin_w, args.bin_h), dtype=int))
            item_state2[0:width, 0:height] = 0
            next_state = torch.cat((bin_state, item_state2), 1).unsqueeze(0)

  
            ''' store experience '''
            sequence_count += 1
            total_steps += 1
            done = True if (sequence_count == (args.max_sequence)) or (torch.sum(bin_state) == 0) else False
            exp =  (state, action, reward, next_state, done)
            replay.append(exp) 


            ''' update model '''
            if len(replay) > args.learn_start:
                loss = agent.learn(replay)
                step_losses.append(loss)

            ''' update target model '''
            if (total_steps % args.sync_freq) == 0:
                agent.update_target_net()

            ''' end of iteration '''
            if (sequence_count == (args.max_sequence)) or (torch.sum(bin_state) == 0):
                ''' update epsilon '''
                if args.epsilon > 0:
                    args.epsilon -= (1/(args.iterations/1.8))
                ''' adjust the learning rate after each iteration '''
                agent.scheduler.step()

                # compute packing efficiency of the final bin state
                PE = torch.sum(1-bin_state) / (args.bin_w * args.bin_h)
                # store the training log
                iteration_losses.append(np.mean(step_losses[-sequence_count:]))
                iteration_rewards.append(rewards)
                sequence_count_list.append(sequence_count)
                PE_list.append(PE)
                break

        ''' print training info '''
        if ((iter + 1) % 5000 == 0):
            print(f"\niter: {iter + 1}, "
                f"PE: {np.mean(PE_list[-20:]):.4f}, "
                f"rewards: {np.mean(iteration_rewards[-20:]):.4f}, "
                f"loss: {np.mean(iteration_losses[-20:]):.4f}")
            
    end_time = time.asctime(time.localtime(time.time()))
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    print(f"Start time: {begin_time}")
    print(f"End time: {end_time}")
    print(f"Finish in: {int(h):02d}:{int(m):02d}:{int(s):02d}")
    

    ''' save model and training info '''
    agent.save(dir_name, f"{args.task}_{args.bin_w}x{args.bin_h}.pt")
    np.savetxt(f"{dir_name}/train/iteration_losses.txt", iteration_losses, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/train/step_rewards.txt", step_rewards, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/train/iteration_rewards.txt", iteration_rewards, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/train/PE_list.txt", PE_list, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/train/sequence_count_list.txt", sequence_count_list, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/train/action_list.txt", action_list, delimiter =", ", fmt ='%s')
    
    ''' save hyper-parameters '''
    with open(f"{dir_name}/hyperparameters.txt", "w") as f:
        f.write(f"Hyper-parameters:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

    # plot loss
    plt.figure(figsize=(10,7))
    plt.plot(iteration_losses ,alpha=0.5)
    plt.plot(utils.moving_average(iteration_losses,50), color="blue")
    plt.title(f"{str(args.task).title()} {str(args.bin_w).title()}x{str(args.bin_h).title()} Iterations Training Loss",fontsize=20)
    plt.xlabel("Iterations",fontsize=18)
    plt.ylabel("Average Loss",fontsize=18)
    plt.savefig(f"{dir_name}/img/iterations_training_loss.png", dpi=300)
    plt.close()

    # plot rewards
    plt.figure(figsize=(10,7))
    plt.plot(iteration_rewards ,alpha=0.5)
    plt.plot(utils.moving_average(iteration_rewards,50), color="blue")
    plt.title(f"{str(args.task).title()} {str(args.bin_w).title()}x{str(args.bin_h).title()} Iterations Training Reward",fontsize=20)
    plt.xlabel("Iterations",fontsize=18)
    plt.ylabel("Cumulative Rewards",fontsize=18)
    plt.savefig(f"{dir_name}/img/iterations_training_rewards.png", dpi=300)
    plt.close()

    # plot PE
    plt.figure(figsize=(10,7))
    plt.plot(PE_list ,alpha=0.5)
    plt.plot(utils.moving_average(PE_list, 50), color="blue")
    plt.title(f"{str(args.task).title()} {str(args.bin_w).title()}x{str(args.bin_h).title()} Iterations Packing Efficiency",fontsize=20)
    plt.xlabel("Iterations",fontsize=18)
    plt.ylabel("Packing Efficiency",fontsize=18)
    plt.savefig(f"{dir_name}/img/iterations_training_PE.png", dpi=300)
    plt.close()
    