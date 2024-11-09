
## Run an experiment 
To run 6a and 9a we need to install particle environment from "https://github.com/thu-rllab/CFCQL/tree/main/continuous/multiagent-particle-envs" 

```shell
python3 src/main.py --config=offpg_smac --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a t_max=2000000
```
