REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_agent import ModularGatedCascadeCondNet
from .grnn_agent import GRNNAgent
from .depth_route_net import DepthRouteNet

REGISTRY["rnn"] = RNNAgent
REGISTRY["soft"] = ModularGatedCascadeCondNet
REGISTRY["soft_new"] = DepthRouteNet
REGISTRY["grnn"] = GRNNAgent
