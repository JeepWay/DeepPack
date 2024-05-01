import argparse
import time
from matplotlib import pyplot as plt
import utils
import torch
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
import copy

from agent import Agent

def preargparse():
    parser = argparse.ArgumentParser(description='This is a simple program to demonstrate argparse')
    # hyper-parameters mentioned in the paper
    parser.add_argument("--bin_w", default=5, type=int, help="width of bin", choices=[3, 4, 5])
    parser.add_argument("--bin_h", default=5, type=int, help="height of bin", choices=[3, 4, 5])
    parser.add_argument("--task", default="square", type=str, help="task name", choices=["unit_square", "rectangular", "square"])
    parser.add_argument("--learning_rate", default=1e-3, type=int)
    parser.add_argument("--gamma", default=0.95, type=int)
    parser.add_argument("--mem_size", default=20000, type=int, help="replay buffer size")
    parser.add_argument("--K", default=2, type=int, help="coefficient of bonus PE reward for last step")
    parser.add_argument("--epsilon", default=1., type=float)
    parser.add_argument("--iterations", default=1000, type=int, help="each iteration consists of w*h items")
    # other hyper-parameters
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learn_start", default=2000, type=int)
    parser.add_argument("--sync_freq", default=2000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_sequence", default=1, type=int, help="lens of pre-defined sequence")
    parser.add_argument("--sequence_type", default='random', type=str, help="type of pre-defined sequence", choices=['type1', 'type2', 'type3', 'random'])
    args = parser.parse_args()
    args.action_space = args.bin_w * args.bin_h + 1
    # args.max_sequence = args.bin_w * args.bin_h
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    begin_time = time.asctime(time.localtime(time.time()))
    start = time.time()
    args = preargparse()

    agent = Agent(args)
    dir_name = f"{args.task}_{args.bin_w}x{args.bin_h}_result"
    model_name = f"./{dir_name}/{args.task}_{args.bin_w}x{args.bin_h}.pt"
    agent.net.load_state_dict(torch.load(model_name))
    agent.eval()

    step_rewards = []
    iteration_rewards = []
    PE_list = []
    sequence_count_list = []
    action_list = []
    bin_list = []

    ''' read test data '''
    file = open(f"./data/test_{args.task}_{str(args.bin_w)}_{args.sequence_type}_{args.iterations}.txt",'r')
    print("file", file)
    content = file.read()
    file.close()
    items = content.split('\n')
    if items[-1] == "":
        items.pop()


    for iter in tqdm(range(args.iterations)):
        sequence_count = 0
        status = 1
        rewards = 0
        bin_state = torch.from_numpy(np.ones((args.bin_w, args.bin_h), dtype=int))
        if (iter==1):
            bin_list.append(copy.deepcopy(bin_state))

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
            action = agent.act(state.unsqueeze(0).float().to(args.device))
            action_list.append(action)

            ''' calculate reward then update bin_state'''
            if (action != (args.action_space-1)):
                x = action // args.bin_h
                y = action % args.bin_h

                if (x+width > args.bin_w) or (y+height > args.bin_h):
                    reward = -5
                elif (torch.sum(bin_state[x:x+width, y:y+height]) != (width * height)):
                    reward = -5
                else:
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
            else:
                reward = 0
            # additional reward for last step
            if (sequence_count+1 == (args.max_sequence)) or (torch.sum(bin_state) == 0):
                PE = torch.sum(1-bin_state) / (args.bin_w * args.bin_h)
                reward += args.K * PE
            step_rewards.append(reward)
            rewards += reward
            if (iter==1):
                bin_list.append(copy.deepcopy(bin_state))

            ''' get next item and state of next item '''
            if (iter * args.max_sequence + sequence_count + 1) == len(items):
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

            ''' update model '''

            ''' update target model '''

            ''' update epsilon '''

            ''' end of iteration '''
            if (sequence_count == (args.max_sequence)) or (torch.sum(bin_state) == 0):
                # compute packing efficiency
                PE = torch.sum(1-bin_state) / (args.bin_w * args.bin_h)
                #print(f"iter: {iter}, sequence_count: {sequence_count}, PE: {PE:.3f} rewards: {rewards:.3f}, loss: {np.mean(step_losses[-sequence_count:]):.4f}")
                iteration_rewards.append(rewards)
                sequence_count_list.append(sequence_count)
                PE_list.append(PE)
                break

        ''' print testing info '''
        if ((iter + 1) % 5000 == 0):
            print(f"\niter: {iter + 1}, "
                f"PE: {np.mean(PE_list[-2000:]):.4f}, "
                f"rewards: {np.mean(iteration_rewards[-2000:]):.4f}")

    end_time = time.asctime(time.localtime(time.time()))
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    print(f"Start time: {begin_time}")
    print(f"End time: {end_time}")
    print(f"Finish in: {int(h):02d}:{int(m):02d}:{int(s):02d}")


    ''' save model and testing info '''
    np.savetxt(f"{dir_name}/test/step_rewards.txt", step_rewards, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/test/iteration_rewards.txt", iteration_rewards, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/test/PE_list.txt", PE_list, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/test/sequence_count_list.txt", sequence_count_list, delimiter =", ", fmt ='%s')
    np.savetxt(f"{dir_name}/test/action_list.txt", action_list, delimiter =", ", fmt ='%s')

    # plot rewards
    plt.figure(figsize=(10,7))
    plt.plot(iteration_rewards ,alpha=0.5)
    plt.plot(utils.moving_average(iteration_rewards,50), color="blue")
    plt.title(f"{str(args.task).title()} {str(args.bin_w).title()}x{str(args.bin_h).title()} Testing Reward",fontsize=20)
    plt.xlabel("Iterations",fontsize=18)
    plt.ylabel("Cumulative Rewards",fontsize=18)
    plt.savefig(f"{dir_name}/img/{args.sequence_type}_testing_rewards.png", dpi=300)
    plt.close()

    # plot PE
    plt.figure(figsize=(10,7))
    plt.plot(PE_list ,alpha=0.5)
    plt.plot(utils.moving_average(PE_list, 50), color="blue")
    plt.title(f"{str(args.task).title()} {str(args.bin_w).title()}x{str(args.bin_h).title()} Packing Efficiency",fontsize=20)
    plt.xlabel("Iterations",fontsize=18)
    plt.ylabel("Packing Efficiency",fontsize=18)
    plt.savefig(f"{dir_name}/img/{args.sequence_type}_testing_PE.png", dpi=300)
    plt.close()

    ''' plot bin images '''
    for i in range(len(bin_list)):
        utils.bin2image(1-bin_list[i])
        import os
        if not os.path.isdir(f"{dir_name}/img/{args.sequence_type}"):
            os.mkdir(f"{dir_name}/img/{args.sequence_type}")
        plt.savefig(f"{dir_name}/img/{args.sequence_type}/{i}.png")
        plt.close()
        
    ''' save bin images to gif '''
    utils.images_to_gif(f"{dir_name}/img/{args.sequence_type}", 
                        f"{dir_name}/img/test_{args.sequence_type}.gif")

    ''' save testing info '''
    with open(f"{dir_name}/test/test_result.txt", "a") as f:
        f.write(f"Sequence type: {args.sequence_type}\n")
        f.write(f"Average PE: {np.mean(PE_list):.4f}\n")
        f.write(f"Std PE: {np.std(PE_list):.4f}\n")
        f.write(f"Average rewards: {np.mean(iteration_rewards):.4f}\n\n")
    print(f"Average PE: {np.mean(PE_list):.4f}, "
          f"Std PE: {np.std(PE_list):.4f}\n"
        f"Average rewards: {np.mean(iteration_rewards):.4f}\n")
    

