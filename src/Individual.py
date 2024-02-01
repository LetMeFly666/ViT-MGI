class Individual:
    def __init__(self, id) -> None:
        self.id = id
        self.b = 0.9      # belive
        self.d = 0      # disbelive
        self.u = 0.1      # updates
        self.a = 0.5

    def get_reputation_value(self):
        return self.b + self.u * self.a
    
    def update_param(self, new_b, new_d, new_u):
        self.b = new_b
        self.d = new_d
        self.u = new_u
        
    def __repr__(self) -> str:
        return f"Individual(id: {self.id}, b: {self.b}, d: {self.d}, u: {self.u})"


    


