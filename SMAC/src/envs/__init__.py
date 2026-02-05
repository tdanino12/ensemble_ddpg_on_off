from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .stag_hunt import StagHunt
from .petting_zoo import Pursuit
from .lbforaging import ForagingEnv
import sys
from .macf import MultiAgentCar
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["pursuit"] = partial(env_fn, env=Pursuit)
REGISTRY["foraging"] =  partial(env_fn, env=ForagingEnv)
REGISTRY["macf"] = partial(env_fn, env=MultiAgentCar)
if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
