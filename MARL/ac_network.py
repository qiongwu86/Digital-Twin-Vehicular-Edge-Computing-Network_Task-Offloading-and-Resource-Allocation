import torch
import torch.nn.functional as F

class actor_network(torch.nn.Module):
    def __init__(self, arg):
        super(actor_network,self).__init__()
        self.fc1 = torch.nn.Linear(arg.n_features, arg.n_hidden_1)
        self.fc2 = torch.nn.Linear(arg.n_hidden_1, arg.n_hidden_2)
        self.fc3 = torch.nn.Linear(arg.n_hidden_2, arg.actor_hidden)
        self.action_out = torch.nn.Linear(arg.actor_hidden, arg.n_output)
        self.action_bound = arg.action_bound
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))*self.action_bound
        actions = F.softmax(self.action_out(x),dim=-1)
        #actions = self.action_bound * F.tanh(self.action_out(x))
        return actions.to(torch.float16)

class crtic_network(torch.nn.Module):
    def __init__(self, arg):
        super(crtic_network,self).__init__()
        self.fc1 = torch.nn.Linear(arg.state_dim + arg.action_dim, arg.n_hidden_1)
        self.fc2 = torch.nn.Linear(arg.n_hidden_1, arg.n_hidden_2)
        self.fc3 = torch.nn.Linear(arg.n_hidden_2, arg.critic_hidden)
        self.fc4 = torch.nn.Linear(arg.critic_hidden, 1)
        self.action_bound = arg.action_bound
    
    def forward(self, x, a):
        x = torch.cat(x, dim=1)
        for i in range(len(a)):
            a[i] /= self.action_bound
        a = torch.cat(a, dim=1)
        cat = torch.cat([x,a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        q_value = self.fc4(x)
        return q_value