class Space(object):
    def sample(self):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

    def to_jsonable(self, sample_n):
        return sample_n

    def from_jsonable(self, sample_n):
        return sample_n
