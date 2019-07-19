class VecEnv(object):
    def step(self, vac):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        pass
