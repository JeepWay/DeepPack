import os
import torch
import numpy as np
import random

def makeTestDir(task, bin_w, bin_h, epochs):
    print(f"GPU: {torch.cuda.is_available()}, {torch.cuda.get_device_name()}")

    import datetime
    date_string = str(datetime.date.today())
    dir_name = f"./{task}_{bin_w}x{bin_h}_result/"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        os.mkdir(dir_name + "train/")
        os.mkdir(dir_name + "test/")
        os.mkdir(dir_name + "img/")
    else:
        print(dir_name + " is exist")

    return dir_name, date_string


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


def deleteFile(dir_name, second_dir_name, filename):
    try:
        os.remove(f"{dir_name}{second_dir_name}{filename}")
    except OSError as e:
        print(e)
    else:
        print(f"File is deleted successfully: {filename}")


def deleteLastOutput(dir_name):
    deleteFile(dir_name, "train/", "iteration_losses.txt")
    deleteFile(dir_name, "train/", "step_rewards.txt")
    deleteFile(dir_name, "train/", "iteration_rewards.txt")
    deleteFile(dir_name, "train/", "PE_list.txt")
    deleteFile(dir_name, "train/", "sequence_count_list.txt")
    deleteFile(dir_name, "train/", "action_list.txt")


def moving_average(lst, move): 
    moving_averages = []
    for i in range(len(lst)):
        start_idx = max(0, i - move)
        end_idx = min(i + move + 1, len(lst))
        window = lst[start_idx:end_idx]
        average = sum(window) / len(window)
        moving_averages.append(average)

    return moving_averages


def bin2image(state, x, y, item_w, item_h):
    import matplotlib.pyplot as plt
    plt.imshow(state, cmap='gray', origin='upper', vmin=0, vmax=1)
    plt.title(f'{state.shape[0]}x{state.shape[1]} bin', fontsize=20)
    plt.xlabel('height', fontsize=16)
    plt.ylabel('width', fontsize=16)
    plt.scatter(-0.5+y, -0.5+x, s=250, color='red', marker='o')
    plt.text(-0.5, -1, f'item:{item_w}x{item_h}', fontsize=18, color='red', ha='center', va='center')
    plt.xticks(np.arange(-0.5, state.shape[1]), np.arange(0, state.shape[1]+1))
    plt.yticks(np.arange(-0.5, state.shape[0]), np.arange(0, state.shape[0]+1))
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=2)

 
def images_to_gif(image_folder, gif_path, fps=1):
    import imageio
    # Get the list of image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images = sorted(images, key=lambda x: int(x.split('.')[0]))
    # Create frames list
    frames = []
    for image in images:
        image_path = os.path.join(image_folder, image)
        frames.append(imageio.v2.imread(image_path))
    # Save as gif
    imageio.mimsave(gif_path, frames, fps=0.6, loop=0, loop_delay=7)
    

if __name__ == '__main__':
    images_to_gif("./square_5x5_result/img/type2", "./square_5x5_result/type2.gif", fps=1)

    