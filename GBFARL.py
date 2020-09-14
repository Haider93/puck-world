

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy
import scipy.stats
import random
from agent import ApproxQAgent
from agent import SarsaAgent
import matplotlib.pyplot as plt

RAD2DEG = 57.29577951308232  


class PuckWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, action_dist = []):
        self.width = 600  #  screen width
        self.height = 600  
        self.l_unit = 1.0  # pysical world width
        self.v_unit = 1.0  # velocity
        self.max_speed = 0.025  # max agent velocity along a axis

        self.re_pos_interval = 30  
        self.accel = 0.002  
        self.rad = 0.05  # agent 
        self.target_rad = 0.01  # target radius.
        self.goal_dis = self.rad  # expected goal distance
        self.t = 0  # puck world clock
        self.update_time = 100  # time for target randomize its position
        self.low = np.array([0,  # agent position x
                             0,
                             -self.max_speed,  # agent velocity
                             -self.max_speed,
                             0,  # target position x
                             0,
                             ])
        self.high = np.array([self.l_unit,
                              self.l_unit,
                              self.max_speed,
                              self.max_speed,
                              self.l_unit,
                              self.l_unit,
                              ])
        self.reward = 0  # for rendering
        self.action = None  # for rendering
        self.viewer = None
        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Discrete(5)
        self.action_dist = action_dist
        self.action_observation = np.zeros((5,), dtype=int)
        random.seed(10)
        self.alpha = 0.001
        self.gamma = 0.975
        self.epsilon = 0.7
        self.vals = []
        #self.action_space = np.random.normal(0.0, 1.0, size=None)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self):
        # assert self.action_space.contains(action), \
        #     "%r (%s) invalid" % (action, type(action))

        #self.action = action  # action for rendering
        self.prev_state = copy.copy(self.state)
        normalized_prev_state = self.normalized_state(np.array(self.prev_state))
        self.action = self.pick_action(self.epsilon, self.action_dist,normalized_prev_state)
        action = self.action
        ppx, ppy, pvx, pvy, tx, ty = self.state 
        ppx, ppy = ppx + pvx, ppy + pvy  # update agent position
        pvx, pvy = pvx * 0.95, pvy * 0.95  # natural velocity loss

        if action == 0: pvx -= self.accel  # left
        if action == 1: pvx += self.accel  # right
        if action == 2: pvy += self.accel  # up
        if action == 3: pvy -= self.accel  # down
        if action == 4: pass  # no move

        if ppx < self.rad:  # encounter left bound
            pvx *= -0.5
            ppx = self.rad
        if ppx > 1 - self.rad:  # right bound
            pvx *= -0.5
            ppx = 1 - self.rad
        if ppy < self.rad:  # bottom bound
            pvy *= -0.5
            ppy = self.rad
        if ppy > 1 - self.rad:  # right bound
            pvy *= -0.5
            ppy = 1 - self.rad

        self.t += 1
        if self.t % self.update_time == 0:  # update target position
            tx = self._random_pos()  # randomly
            ty = self._random_pos()

        dx, dy = ppx - tx, ppy - ty  # calculate distance from
        dis = self._compute_dis(dx, dy)  # agent to target

        self.reward = self.goal_dis - dis  # give an reward

        done = bool(dis <= self.goal_dis)

        self.state = (ppx, ppy, pvx, pvy, tx, ty)
        return self.normalized_state(np.array(self.state)), self.normalized_state(np.array(self.prev_state)), self.reward, self.action, dis, done
        #return np.array(self.state), np.array(self.prev_state), self.reward, self.action, done, {}

    def _step_b(self, action):
        # assert self.action_space.contains(action), \
        #     "%r (%s) invalid" % (action, type(action))

        self.action = action  # action for rendering
        ppx, ppy, pvx, pvy, tx, ty = self.state  
        ppx, ppy = ppx + pvx, ppy + pvy  # update agent position
        pvx, pvy = pvx * 0.95, pvy * 0.95  # natural velocity loss

        if action == 0: pvx -= self.accel  # left
        if action == 1: pvx += self.accel  # right
        if action == 2: pvy += self.accel  # up
        if action == 3: pvy -= self.accel  # down
        if action == 4: pass  # no move

        if ppx < self.rad:  # encounter left bound
            pvx *= -0.5
            ppx = self.rad
        if ppx > 1 - self.rad:  # right bound
            pvx *= -0.5
            ppx = 1 - self.rad
        if ppy < self.rad:  # bottom bound
            pvy *= -0.5
            ppy = self.rad
        if ppy > 1 - self.rad:  # right bound
            pvy *= -0.5
            ppy = 1 - self.rad

        self.t += 1
        if self.t % self.update_time == 0:  # update target position
            tx = self._random_pos()  # randomly
            ty = self._random_pos()

        dx, dy = ppx - tx, ppy - ty  # calculate distance from
        dis = self._compute_dis(dx, dy)  # agent to target

        self.reward = self.goal_dis - dis  # give an reward

        done = bool(dis <= self.goal_dis)

        self.state = (ppx, ppy, pvx, pvy, tx, ty)
        return np.array(self.state), self.reward, done, dis

    def _random_pos(self):
        return self.np_random.uniform(low=0, high=self.l_unit)

    def _compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

    def _reset(self):
        self.state = np.array([self._random_pos(),
                               self._random_pos(),
                               0,
                               0,
                               self._random_pos(),
                               self._random_pos()
                               ])
        return self.state  # np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scale = self.width / self.l_unit 
        rad = self.rad * scale  
        t_rad = self.target_rad * scale  # target radius

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)


            target = rendering.make_circle(t_rad, 30, True)
            target.set_color(0.1, 0.9, 0.1)
            self.viewer.add_geom(target)
            target_circle = rendering.make_circle(t_rad, 30, False)
            target_circle.set_color(0, 0, 0)
            self.viewer.add_geom(target_circle)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target_circle.add_attr(self.target_trans)

            self.agent = rendering.make_circle(rad, 30, True)
            self.agent.set_color(0, 1, 0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            agent_circle = rendering.make_circle(rad, 30, False)
            agent_circle.set_color(0, 0, 0)
            agent_circle.add_attr(self.agent_trans)
            self.viewer.add_geom(agent_circle)

            # start_p = (0, 0)
            # end_p = (0.7 * rad, 0)
            # self.line = rendering.Line(start_p, end_p)
            # self.line.linewidth = rad / 10
            self.line_trans = rendering.Transform()
            # self.line.add_attr(self.line_trans)
            # self.viewer.add_geom(self.line)
            self.arrow = rendering.FilledPolygon([
                (0.7 * rad, 0.15 * rad),
                (rad, 0),
                (0.7 * rad, -0.15 * rad)
            ])
            self.arrow.set_color(0, 0, 0)
            self.arrow.add_attr(self.line_trans)
            self.viewer.add_geom(self.arrow)

    
        ppx, ppy, _, _, tx, ty = self.state
        self.target_trans.set_translation(tx * scale, ty * scale)
        self.agent_trans.set_translation(ppx * scale, ppy * scale)
        vv, ms = self.reward + 0.3, 1
        r, g, b, = 0, 1, 0
        if vv >= 0:
            r, g, b = 1 - ms * vv, 1, 1 - ms * vv
        else:
            r, g, b = 1, 1 + ms * vv, 1 + ms * vv
        self.agent.set_color(r, g, b)

        a = self.action
        if a in [0, 1, 2, 3]:
            degree = 0
            if a == 0:
                degree = 180
            elif a == 1:
                degree = 0
            elif a == 2:
                degree = 90
            else:
                degree = 270
            self.line_trans.set_translation(ppx * scale, ppy * scale)
            self.line_trans.set_rotation(degree / RAD2DEG)
            # self.line.set_color(0,0,0)
            self.arrow.set_color(0, 0, 0)
        else:
            # self.line.set_color(r,g,b)
            self.arrow.set_color(r, g, b)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def pick_action(self, epsilon, action_dist, state):
        if(random.uniform(0, 1) < epsilon):
            return random.randint(0, 4)
        action = 0
        max = action_dist[action].pdf(state)
        for i in range(1, 5):
            action_value = action_dist[i].pdf(state)
            if max < action_value:
                max = action_value
                action = i
        return action

    def getQMax(self, action_dist, state):
        action = 0
        max = action_dist[action].pdf(state)
        for i in range(1, 5):
            action_value = action_dist[i].pdf(state)
            if max < action_value:
                max = action_value
                action = i
        return action

    def getQ(self, state, action):
        return self.action_dist[action].pdf(state)

    def normalized_state(self, state):
        sum = 0.0
        for i in range(0, len(state)):
            sum = sum + (1/5.0 * state[i])
        return sum

    def plotcur(self, val, action_dist):
        self.vals.append(val)
        if len(self.vals) == 100:
            x = np.asarray(self.vals)
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[0].mean(), action_dist[0].std()), 'ro')
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[1].mean(), action_dist[1].std()), 'g^')
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[2].mean(), action_dist[2].std()), 'g-')
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[3].mean(), action_dist[3].std()), 'g+')
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[4].mean(), action_dist[4].std()), 'go')
            #plt.plot(action_dist[0].mean(), action_dist[0].std())
            plt.show()

    def TD_learning(self, new_state, old_state, reward, action):
        prev_obs = self.action_observation[action]
        self.action_observation[action] = self.action_observation[action] + 1
        old_mean = self.action_dist[action].mean()
        #update = reward + self.gamma * self.getQ(new_state, self.pick_action(self.epsilon, self.action_dist, new_state)) - self.getQ(old_state, action)
        update = reward + self.gamma * self.getQMax(self.action_dist,new_state) - self.getQ(old_state, action)
        update = self.alpha * update
        new_mean = old_mean + ((update - old_mean)/self.action_observation[action])
        old_var = self.action_dist[action].var()
        new_var = (prev_obs / (prev_obs+1)) * (old_var + (pow((update - old_mean), 2.0) / (prev_obs + 1)))
        if new_var<0:
            exit(-1)
        if new_var == 0:
            new_var = old_var
        self.action_dist[action] = scipy.stats.norm(new_mean, math.sqrt(new_var))
        #self.plotcur(update, self.action_dist)

    def TD_learning_mvd(self, new_state, old_state, reward, action):
        prev_obs = self.action_observation[action]
        self.action_observation[action] = self.action_observation[action] + 1
        old_mean = self.action_dist[action].mean
        #update = reward + self.gamma * self.getQ(new_state, self.pick_action(self.epsilon, self.action_dist, new_state)) - self.getQ(old_state, action)
        update = reward + self.gamma * self.getQMax(self.action_dist,new_state) - self.getQ(old_state, action)
        update = self.alpha * update
        new_mean = old_mean + ((update - old_mean)/self.action_observation[action])
        old_var = self.action_dist[action].cov
        new_var = (prev_obs / (prev_obs+1)) * (old_var + (pow((update - old_mean), 2.0) / (prev_obs + 1)))
        if np.linalg.det(new_var)<0:
            exit(-1)
        if np.linalg.det(new_var) == 0:
            new_var = old_var
        self.action_dist[action] = scipy.stats.multivariate_normal(new_mean, new_var)
        #self.plotcur(update, self.action_dist)
if __name__ == "__main__":
    action_dist = []
    # action_dist.append(scipy.stats.norm(0.0, 1.0))
    # action_dist.append(scipy.stats.norm(0.0, 1.0))
    # action_dist.append(scipy.stats.norm(0.0, 1.0))
    # action_dist.append(scipy.stats.norm(0.0, 1.0))
    # action_dist.append(scipy.stats.norm(0.0, 1.0))

    ##multivariate
    state_dimen = 6
    mvd = scipy.stats.multivariate_normal(np.zeros(state_dimen), np.eye(state_dimen, state_dimen))
    action_dist.append(mvd)
    action_dist.append(mvd)
    action_dist.append(mvd)
    action_dist.append(mvd)
    action_dist.append(mvd)

    env = PuckWorldEnv(action_dist)

    ##DQN AGENT
    # agent = ApproxQAgent(env,
    #                      trans_capacity=50000,
    #                      hidden_dim=32)
    # # env._reset()
    # agent.learning(gamma=0.975,
    #                learning_rate=0.001,
    #                batch_size=64,
    #                max_episodes=5, 
    #                min_epsilon=0.2,  
    #                epsilon_factor=0.7,  
    #                # Episodes,
    #                # min_epsilon,
    #                epochs=2  
    #                )

    ##TILE-CODINGS
    # env._reset()
    # agent = SarsaAgent(env)
    # agent.learning(gamma=0.80,
    #              alpha=0.001,
    #                max_num_episode=1)


    # nfs = env.observation_space.shape[0]
    # nfa = env.action_space
    # print("nfs:%s; nfa:d" % (nfs))
    # print(env.observation_space)
    # print(env.action_space)
    env._reset()
    file = open('TD_Q_3.csv', 'w')
    file.write("Steps in Episode" + "," + "Distance from goal" + "\n")

#     file = open('GBNLFA.csv', 'w')
#     file.write("Episode"+","+"reward"+"\n")
    done = False
    tot_reward = 0
    for i in range(5):
        while not done:
           env._render()
           #env._step_b(env.action_space.sample())
           state, prev_state, reward, action, dis_info, done = env._step()
           tot_reward += reward
           #univariate
           #env.TD_learning(state, prev_state, reward, action)
           ##multivariate
           env.TD_learning_mvd(state, prev_state, reward, action)
           print("Step in episode :: ",i)
           print("Distance from goal :: ", dis_info)
           file.write(str(i)+","+str(tot_reward)+"\n")
    file.close()   
    print("env closed")
