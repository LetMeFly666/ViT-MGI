class Individual:
    def __init__(self, id) -> None:
        self.id = id
        self.b = 0.9      # belive
        self.d = 0        # disbelive
        self.u = 0.1      # uncerntainty
        self.a = 0.5
        self.z = 0.6      # a given fade parameter about event freshness
        self.T = []

    def get_reputation_value(self): 
        ret = self.z * (self.b + self.u * self.a)
        sum = self.z
        Y = len(self.T)  # 加上历史信誉的影响
        for idx, repu in enumerate(self.T):  # BUMP!
            ret += self.z ** (Y - (idx + 1)) * repu
            sum += self.z ** (Y - (idx + 1))
        return ret / sum
    
    def update_param(self, new_b, new_d, new_u):
        self.T.append(self.b + self.u * self.a) # 记录历史信誉值
        self.b = new_b
        self.d = new_d
        self.u = new_u
        
    def __repr__(self) -> str:
        return f"Individual(id: {self.id:>2d}, b: {self.b:4.2f}, d: {self.d:4.2f}, u: {self.u:4.2f}, a: {self.a:4.2f})"
