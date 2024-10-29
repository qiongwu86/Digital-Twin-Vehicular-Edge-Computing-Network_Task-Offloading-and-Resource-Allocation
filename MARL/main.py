from marl_train import Runner, Agent
from setting import arg
import my_env

if __name__ == '__main__':
    # get the params
    args = arg()
    env = my_env.Environment(args.lane_num, args.n_agents, args.width, args.task_num)
    runner = Runner(args, env)
    if args.evaluate:
        pass
    else:
        runner.run()