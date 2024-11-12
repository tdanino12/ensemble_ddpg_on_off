import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PGCriticNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, args):
        super(PGCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class FACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        #self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        #self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        #self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)


        self.network1 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network2 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network3 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network4 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network5 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network6 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network7 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network8 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network9 = PGCriticNetwork(self.input_shape, self.n_actions, args)
        self.network10 = PGCriticNetwork(self.input_shape, self.n_actions, args) 
        self.network11 = PGCriticNetwork(self.input_shape, self.n_actions, args) 
        self.network12 = PGCriticNetwork(self.input_shape, self.n_actions, args) 
        self.network13 = PGCriticNetwork(self.input_shape, self.n_actions, args) 
        self.network14 = PGCriticNetwork(self.input_shape, self.n_actions, args) 


    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        #x = F.relu(self.fc1(inputs))
        #x = F.relu(self.fc2(x))
        #q = self.fc3(x)
        
        q = self.network1(inputs)
        q2 = self.network2(inputs)
        q3 = self.network3(inputs)
        q4 = self.network4(inputs)
        q5 = self.network5(inputs)
        q6 = self.network6(inputs)
        q7 = self.network7(inputs)
        q8 = self.network8(inputs)
        q9 = self.network9(inputs)
        q10 = self.network10(inputs)  


        q11 = self.network7(inputs)
        q12 = self.network8(inputs)
        q13 = self.network9(inputs)
        q14 = self.network10(inputs)
        

        #return (q+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14)/th.tensor(14),q,q2,q3,q4,q5,q6,q7,q8,q9,q10
        return (q+q2+q3+q4+q5+q6+q7+q8+q9+q10+q11+q12+q13+q14)/th.tensor(14), hidden_state#,q,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14        
        
        
        
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape
