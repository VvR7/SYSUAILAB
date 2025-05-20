import argparse
import gym
from argument import dqn_arguments, pg_arguments
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    args, _ = parser.parse_known_args()
    if args.train_dqn:
        parser = dqn_arguments(parser)
    elif args.train_pg:
        parser = pg_arguments(parser)
    
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        pass

    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        return_list = agent.train()  # 调用train方法开始训练
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.axhline(y=200, color='orange', linestyle='-',label='200')
        plt.axhline(y=180, color='red', linestyle='-',label='180')
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title(f'{args.dqn_type} on {env_name}')
        plt.savefig('result.png')
        plt.show()


if __name__ == '__main__':
    args = parse()
    run(args)
