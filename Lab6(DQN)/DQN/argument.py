def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--gamma", default=0.98, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)
    
    # 添加DQN需要的额外参数
    parser.add_argument("--epsilon", default=0.01, type=float, help='epsilon for epsilon-greedy exploration')
    parser.add_argument("--target_update", default=10, type=int, help='update frequency of target network')
    parser.add_argument("--batch_size", default=64, type=int, help='batch size for training')
    parser.add_argument("--buffer_size", default=10000, type=int, help='replay buffer size')
    parser.add_argument("--dqn_type", default="DQN", type=str, help='DQN type: DQN or DoubleDQN')
    parser.add_argument("--num_episodes", default=300, type=int, help='number of episodes to train')
    parser.add_argument("--min_size", default=500, type=int, help='minimum size of replay buffer before training')

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser
