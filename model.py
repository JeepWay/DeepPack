import torch
import torch.nn as nn

class doubleDQN(nn.Module):
    def __init__(self, args):
        super(doubleDQN, self).__init__()
        self.width = args.bin_w
        self.height = 2 * args.bin_h
        self.action_space = args.action_space
        if args.bin_w == 4:
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(1,3), stride=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(1,3), stride=1)
            self.linear_input = 32 * (((((self.width-1)+1)//2)-1)+1) * (((((self.height-3)+1)//2)-3)+1)  
        elif args.bin_w == 5:
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(2,3), stride=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(2,3), stride=1)
            self.linear_input = 32 * (((((self.width-2)+1)//2)-2)+1) * (((((self.height-3)+1)//2)-3)+1)
           
        
        self.model = torch.nn.Sequential(
            self.conv1,
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2), stride=(2,2)),

            self.conv2,
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(self.linear_input, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_space)
        )

    def forward(self, state):
        return self.model(state)





