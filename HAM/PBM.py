import torch
import asset

torch.set_default_dtype(torch.float32)

def make_C_matrix(C_room, C_wall):
    C1, C2 = torch.diag(C_room/20), torch.diag(C_wall/20)
    return C1, C2

def make_D_matrix(R_internal, R_partWall):
    D = torch.zeros((R_internal.shape[1], R_internal.shape[1]))
    R_inWall = R_internal[2]
    R_room = R_internal[3]
    for i in range(R_internal.shape[1]):
        for j in range(R_internal.shape[1]):
            if i == j:
                D[i, j] = -(R_room[i]*R_inWall[i])/(R_room[i]+R_inWall[i])  - torch.sum(R_partWall[i])
            else:
                D[i, j] = R_partWall[i, j]
    
    return D

def make_E_F_matrices(R_internal):
    E = torch.zeros((R_internal.shape[1], R_internal.shape[1]))
    F = torch.zeros((R_internal.shape[1], R_internal.shape[1]))
    R_inWall = R_internal[2]
    R_room = R_internal[3]

    for i in range(R_internal.shape[1]):
        k_2i = (R_room[i]*R_inWall[i])/(R_room[i]+R_inWall[i])
        E[i, i] = k_2i
        F[i, i] = -k_2i
    
    return E, F 

def make_G_matrix(R_internal):
    G = torch.zeros((R_internal.shape[1], R_internal.shape[1]))
    R_ext = R_internal[0]
    R_outwall = R_internal[1]
    R_inWall = R_internal[2]
    R_room = R_internal[3]

    for i in range(R_internal.shape[1]):
        k_1i = (R_ext[i]*R_outwall[i])/(R_ext[i]+R_outwall[i])
        k_2i = (R_room[i]*R_inWall[i])/(R_room[i]+R_inWall[i])
        G[i, i] = -k_2i - k_1i

    return G

class thermoPBM(torch.nn.Module):
    def __init__(self, asset):
        super(thermoPBM, self).__init__()
        self.asset = asset
        R_partwall = torch.tensor(asset.get_R_partWall(), dtype=torch.float32)
        R_internal = torch.tensor(asset.get_R(), dtype=torch.float32)
        C_room, C_wall = asset.get_C()
        C_room, C_wall = torch.tensor(C_room, dtype=torch.float32), torch.tensor(C_wall, dtype=torch.float32)

        self.R_ext = R_internal[0]
        self.R_outwall = R_internal[1]
        self.R_inWall = R_internal[2]
        self.R_room = R_internal[3]

        self.k_1 = R_internal[0] * R_internal[1] / (R_internal[0] + R_internal[1])
        self.k_2 = R_internal[2] * R_internal[3] / (R_internal[2] + R_internal[3])

        self.C1, self.C2 = make_C_matrix(C_room, C_wall)
        self.D = make_D_matrix(R_internal, R_partwall)
        self.E, self.F = make_E_F_matrices(R_internal)
        self.G = make_G_matrix(R_internal)

    def calculate_rhs(self, T_room, T_wall, T_ext, corrective_source_term=torch.zeros(1, 26)):
        u_room, u_wall = self.calculate_U(T_ext)
        rhs_room = torch.matmul(self.D, T_room.T) + torch.matmul(self.E, T_wall.T) + u_room.T + corrective_source_term[:, :13].T
        rhs_wall = torch.matmul(self.F, T_room.T) + torch.matmul(self.G, T_wall.T) + u_wall.T + corrective_source_term[:, 13:].T
        return rhs_room, rhs_wall
    
    def calculate_T_dot(self, T_room, T_wall, T_ext, corrective_source_term=torch.zeros(1, 26)):
        rhs_room, rhs_wall = self.calculate_rhs(T_room, T_wall, T_ext, corrective_source_term)
        T_dot_room = torch.matmul(torch.inverse(self.C1), rhs_room)
        T_dot_wall = torch.matmul(torch.inverse(self.C2), rhs_wall)
        return T_dot_room, T_dot_wall
    
    def calculate_U(self, T_ext, Q_sunExt=torch.zeros(1, 13), Q_sunPen=torch.zeros(1, 13), Q_ir=torch.zeros(1, 13),
                    Ww=0.5, Lr=torch.zeros(1, 13), Wr=0.5, Lc=torch.zeros(1, 13), H=torch.zeros(1, 13), C=torch.zeros(1, 13)):
        # u_room = torch.divide(self.k_2, self.R_inWall)*(Q_sunPen*Ww+Lr) + Q_sunPen*Wr+Lc+H-C
        u_wall = torch.divide(self.k_2, self.R_room)*(Q_sunPen*Wr+Lr) + torch.divide(self.k_1, self.R_ext)*(self.R_ext*T_ext + Q_sunExt + Q_ir)
        u_room = torch.full_like(u_wall, 9.)
        u_room[:, 11] = 30.
        return u_room, u_wall
    
    def step(self, T_room, T_wall, T_ext, delta_t, corrective_source_term=torch.zeros(1, 26)):
        T_dot_room, T_dot_wall = self.calculate_T_dot(T_room, T_wall, T_ext, corrective_source_term)
        T_room_new = torch.add(T_room, delta_t * T_dot_room.T).requires_grad_(True)
        T_wall_new = torch.add(T_wall, delta_t * T_dot_wall.T).requires_grad_(True)
        return T_room_new, T_wall_new
    
    def forward(self, T_room, T_wall, T_ext, delta_t, corrective_source_term=torch.zeros(1, 26)):
        return self.step(T_room, T_wall, T_ext, delta_t, corrective_source_term)


if __name__ == "__main__":
    sim_asset = asset.get_asset()
    thermoPBM = thermoPBM(sim_asset)

    Q_sunPen = torch.zeros(13)
    Q_sunExt = torch.zeros(13)
    Q_ir = torch.zeros(13)
    Lr = torch.zeros(13)
    Lc = torch.zeros(13)
    H = torch.zeros(13)
    C = torch.zeros(13)
    Ww = 0.5
    Wr = 0.5
    T_ext = 30

    delta_t = 60
    N = 1500
    
    T_room = torch.full((13,), 20.)
    T_wall = torch.full((13,), 20.)

    for i in range(N):
        u_room, u_wall = thermoPBM.calculate_U(T_ext, Q_sunExt, Q_sunPen, Q_ir, Ww, Lr, Wr, Lc, H, C)
        print(f'u_room: {u_room}')
        print(f'u_wall: {u_wall}')
        T_room, T_wall = thermoPBM.step(T_room, T_wall, u_room, u_wall, delta_t)
        print(f'T_room: {T_room}')
        print(f'T_wall: {T_wall}')