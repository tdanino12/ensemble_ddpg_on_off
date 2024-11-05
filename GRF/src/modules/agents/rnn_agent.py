import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        '''
        self.fc1_2 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_2 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_3 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_3 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_4 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_4 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_4 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_5 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_5 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_5 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_6 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_6 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_6 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_7 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_7 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_7 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        '''
        '''
        self.fc1_8 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_8 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_8 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_9 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_9 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_9 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc1_10 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_10 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_10 = nn.Linear(args.rnn_hidden_dim, args.n_actions
        '''

    def init_hidden(self):
        x1 = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x2 = self.fc1_2.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x3 = self.fc1_3.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x4 =self.fc1_4.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x5 =self.fc1_5.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x6 =self.fc1_6.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #x7 = self.fc1_7.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x1
        # make hidden states on same device as model
        #return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        #return x1,x2,x3,x4,x5,x6,x7

    def init_hidden2(self):
        x2 = self.fc1_2.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x2

    def init_hidden3(self):
        x3 = self.fc1_3.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x3

    def init_hidden4(self):
        x4 = self.fc1_4.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x4

    def init_hidden5(self):
        x5 = self.fc1_5.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x5

    def init_hidden6(self):
        x6 = self.fc1_6.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x6

    def init_hidden7(self):
        x7 = self.fc1_7.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return x7


    def forward(self, inputs, hidden_state):#,hid2,hid3,hid4,hid5,hid6,hid7):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        '''
        x_2 = F.relu(self.fc1_2(inputs))
        h_in_2 = hid2.reshape(-1, self.args.rnn_hidden_dim)
        h_2 = self.rnn_2(x_2, h_in_2)
        q_2 = self.fc2_2(h_2)

        x_3 = F.relu(self.fc1_3(inputs))
        h_in_3 = hid3.reshape(-1, self.args.rnn_hidden_dim)
        h_3 = self.rnn_3(x_3, h_in_3)
        q_3 = self.fc2_3(h_3)

        x_4 = F.relu(self.fc1_4(inputs))
        h_in_4 = hid4.reshape(-1, self.args.rnn_hidden_dim)
        h_4 = self.rnn_4(x_4, h_in_4)
        q_4 = self.fc2_4(h_4)

        x_5 = F.relu(self.fc1_5(inputs))
        h_in_5 = hid5.reshape(-1, self.args.rnn_hidden_dim)
        h_5 = self.rnn_5(x_5, h_in_5)
        q_5 = self.fc2_5(h_5)

        x_6 = F.relu(self.fc1_6(inputs))
        h_in_6 = hid6.reshape(-1, self.args.rnn_hidden_dim)
        h_6 = self.rnn_6(x_6, h_in_6)
        q_6 = self.fc2_6(h_6)

        x_7 = F.relu(self.fc1_7(inputs))
        h_in_7 = hid7.reshape(-1, self.args.rnn_hidden_dim)
        h_7 = self.rnn_7(x_7, h_in_7)
        q_7 = self.fc2_7(h_7)
        '''
        '''
        x8 = F.relu(self.fc1_8(inputs))
        h_in_8 = hid8.reshape(-1, self.args.rnn_hidden_dim)
        h_8 = self.rnn(x_8, h_in_8)
        q_8 = self.fc8(h_8)

        x9 = F.relu(self.fc1_9(inputs))
        h_in_9 = hid9.reshape(-1, self.args.rnn_hidden_dim)
        h_9 = self.rnn(x_9, h_in_9)
        q_9 = self.fc9(h_9)

        x10 = F.relu(self.fc1_10(inputs))
        h_in_10 = hid10.reshape(-1, self.args.rnn_hidden_dim)
        h_10 = self.rnn(x_10, h_in_10)
        q_10 = self.fc10(h_10)
        '''
        return q,h
        return (q+q_2+q_3+q_4+q_5+q_6+q_7)/th.tensor(7) ,q,q_2,q_3,q_4,q_5,q_6,q_7, h,h_2,h_3,h_4,h_5,h_6,h_7

'''
Soft modularization init function
'''

def _fanin_init(tensor, alpha = 0):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    # bound = 1. / np.sqrt(fan_in)
    bound = np.sqrt( 1. / ( (1 + alpha * alpha ) * fan_in) )
    return tensor.data.uniform_(-bound, bound)

def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)

def _constant_bias_init(tensor, constant = 0.1):
    tensor.data.fill_( constant )

def _normal_init(tensor, mean=0, std =1e-3):
    return tensor.data.normal_(mean,std)

def layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init ):
    weight_init(layer.weight)
    bias_init(layer.bias)

def basic_init(layer):
    layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init)

def uniform_init(layer):
    layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init )

def normal_init(layer):
    layer_init(layer, weight_init = _normal_init, bias_init = _normal_init)
    
def _orthogonal_init(tensor, gain = np.sqrt(2)):
    nn.init.orthogonal_(tensor, gain = gain)

def orthogonal_init(layer, scale = np.sqrt(2), constant = 0 ):
    layer_init(
        layer,
        weight_init= lambda x:_orthogonal_init(x, gain=scale),
        bias_init=lambda x: _constant_bias_init(x, 0))

def null_activation(x):
    return x

'''
Soft modularization base functions
'''

class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func = basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for i, next_shape in enumerate( hidden_shapes ):
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)

            input_shape = next_shape
            self.output_shape = next_shape

    def forward(self, x):

        out = x
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        out = self.last_activation_func(out)
        return out

def calc_next_shape(input_shape, conv_info):
    """
    take input shape per-layer conv-info as input
    """
    out_channels, kernel_size, stride, padding = conv_info
    c, h, w = input_shape
    # for padding, dilation, kernel_size, stride in conv_info:
    h = int((h + 2*padding[0] - ( kernel_size[0] - 1 ) - 1 ) / stride[0] + 1)
    w = int((w + 2*padding[1] - ( kernel_size[1] - 1 ) - 1 ) / stride[1] + 1)
    return (out_channels, h, w )



'''
Soft modularization network
'''
class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, output_shape,
             em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,

            num_layers, num_modules,

            module_hidden,

            gating_hidden, num_gating_layers,

            # gated_hidden
            add_bn = True,
            pre_softmax = False,
            cond_ob = True,
            module_hidden_init_func = basic_init,
            last_init_func = uniform_init,
            activation_func = F.relu,
             **kwargs ):

        super().__init__()

        self.base = MLPBase( 
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = MLPBase(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * \
                                               num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def forward(self, x, embedding_input, return_weights = False):
        # Return weights for visualization
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last(out)

        if return_weights:
            return out, weights, last_weight
        return out

