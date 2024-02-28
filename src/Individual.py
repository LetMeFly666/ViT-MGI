class Individual:
    def __init__(self, id) -> None:
        self.id = id
        self.b = 0.9      # belive
        self.d = 0        # disbelive
        self.u = 0.1      # uncerntainty
        self.a = 0.5

    def get_reputation_value(self):
        return self.b + self.u * self.a
    
    def update_param(self, new_b, new_d, new_u):
        self.b = new_b
        self.d = new_d
        self.u = new_u

    def grow_uncertainty(self):
        if self.d > self.u:
            self.d = self.d - self.u
            self.u += self.u / 2
            self.b += self.u / 2
        
    def __repr__(self) -> str:
        return f"Individual(id: {self.id:>2d}, b: {self.b:4.2f}, d: {self.d:4.2f}, u: {self.u:4.2f}, a: {self.a:4.2f})"


    


