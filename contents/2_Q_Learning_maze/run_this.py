from maze_env import Maze
from RL_brain import QLearningTable
import pandas as pd

def update():
    # 跟着行为轨迹
    df = pd.DataFrame(columns=('state','action_space','reward','Q','action'))
    # 转换为迷宫坐标（x,y）
    def set_state(observation):
        p = []
        p.append(int((observation[0]-5)/40))
        p.append(int((observation[1]-5)/40))
        return p
    for episode in range(100):
        # initial observation
        observation = env.reset()
        observation = set_state(observation)

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            if observation_ != 'terminal':
                observation_ = set_state(observation_)
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            q = RL.q_table.loc[str(observation),action]
            df = df.append(pd.DataFrame({'state':[observation],'action_space':[env.action_space[action]],'reward':[reward],'Q':[q],'action':action}), ignore_index=True)
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    df.to_csv('action.csv')
    RL.q_table.to_csv('q_table.csv')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()