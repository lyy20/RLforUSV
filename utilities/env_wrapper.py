"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe, set_start_method
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines.common.tile_images import tile_images

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # ob, ob_full, reward, done, info = env.step(data)
            ob, reward, done, info = env.step(data)
            if all(done):
                # ob = env.reset()
                pass
            # remote.send((ob, ob_full, reward, done, info))
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            # ob, ob_full = env.reset()
            # remote.send((ob, ob_full))
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        elif cmd == 'A_Star_Info':#返回的信息包括（  智能体数量，障碍物数量，目标数量，智能体基础信息，障碍物基础信息，目标基础信息（位置，大小），智能体朝向角集合 ） !!!需要拿观测值作为障碍物和目标的基础信息，
            agents_info = []
            for agent in env.world.agents:
                agents_info.append([agent.state.p_pos[0],agent.state.p_pos[1],agent.size])
            obstacles_info = []
            for i in env.world.num_obstacles:
                obstacles_info.append([env.world.obstacles[i+env.world.num_obstacles].state.p_pos[0],env.world.obstacles[i+env.world.num_obstacles].state.p_pos[1],env.world.obstacles[i+env.world.num_obstacles].size])
            landmarks_info = []
            for i in env.world.num_landmarks:
                landmarks_info.append([env.world.landmarks[i+env.world.num_landmarks].state.p_pos[0],env.world.landmarks[i+env.world.num_landmarks].state.p_pos[1],env.world.landmarks[i+env.world.num_landmarks].size])
            A_Star_Info = np.concatenate([env.world.num_agents], [env.world.num_obstacles], [env.world.num_landmarks],agents_info,obstacles_info,landmarks_info,env.world.angle)
            remote.send(A_Star_Info)
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        #print('env_wrapper: Process')
        #he possible start methods are 'fork', 'spawn' and 'forkserver'
        # set_start_method('spawn')
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            #print('env_wrapper: p.daemon')    
            p.daemon = True # if the main process crashes, we should not cause things to hang
            #print('env_wrapper: p.start')
            p.start()
        #print('env_wrapper: remote.close')
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        #print('env_wrapper: VecEnv.__init__')
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # obs, obs_full, rews, dones, infos = zip(*results)
        # return np.stack(obs), np.stack(obs_full), np.stack(rews), np.stack(dones), infos
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        # code doesn't work all that well
        # TODO: need to clean up
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def A_star_Info(self):
        for remote in self.remotes:
            remote.send(('A_star_Info', None))
        return np.stack([remote.recv() for remote in self.remotes])


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, obs_full, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done):
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return
