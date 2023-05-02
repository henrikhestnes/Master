import numpy as np
import asset


def make_C_matrix(C_room, C_wall):
    C1, C2 = np.diag(C_room), np.diag(C_wall)
    return C1, C2

def make_D_matrix(R_internal, R_partWall):
    D = np.zeros((R_internal.shape[1], R_internal.shape[1]))
    R_inWall = R_internal[2]
    R_room = R_internal[3]
    for i in range(R_internal.shape[1]):
        for j in range(R_internal.shape[1]):
            if i == j:
                D[i, j] = -(R_room[i]*R_inWall[i])/(R_room[i]+R_inWall[i])  - np.sum(R_partWall[i])
            else:
                D[i, j] = R_partWall[i, j]
    
    return D

def make_E_F_matrices(R_internal):
    E = np.zeros((R_internal.shape[1], R_internal.shape[1]))
    F = np.zeros((R_internal.shape[1], R_internal.shape[1]))
    R_inWall = R_internal[2]
    R_room = R_internal[3]

    for i in range(R_internal.shape[1]):
        k_2i = (R_room[i]*R_inWall[i])/(R_room[i]+R_inWall[i])
        E[i, i] = k_2i
        F[i, i] = -k_2i
    
    return E, F 

def make_G_matrix(R_internal):
    G = np.zeros((R_internal.shape[1], R_internal.shape[1]))
    R_ext = R_internal[0]
    R_outwall = R_internal[1]
    R_inWall = R_internal[2]
    R_room = R_internal[3]

    for i in range(R_internal.shape[1]):
        k_1i = (R_ext[i]*R_outwall[i])/(R_ext[i]+R_outwall[i])
        k_2i = (R_room[i]*R_inWall[i])/(R_room[i]+R_inWall[i])
        G[i, i] = -k_2i - k_1i

    return G

class thermoPBM():
    def __init__(self, asset):
        self.asset = asset
        R_partwall = asset.get_R_partWall()
        R_internal = asset.get_R()
        C_room, C_wall = asset.get_C()

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

    def calculate_rhs(self, T_room, T_wall, u_room, u_wall):
        rhs_room = self.D @ T_room + self.E @ T_wall + u_room
        rhs_wall = self.F @ T_room + self.G @ T_wall + u_wall
        return rhs_room, rhs_wall
    
    def calculate_lhs(self, T_room, T_wall):
        lhs_room = self.C1 @ T_room
        lhs_wall = self.C2 @ T_wall
        return lhs_room, lhs_wall
    
    def calculate_T_dot(self, T_room, T_wall, u_room, u_wall):
        rhs_room, rhs_wall = self.calculate_rhs(T_room, T_wall, u_room, u_wall)
        T_dot_room = np.linalg.inv(self.C1) @ rhs_room
        T_dot_wall = np.linalg.inv(self.C2) @ rhs_wall
        return T_dot_room, T_dot_wall
    
    def calculate_U(self, T_ext, Q_sunExt=np.zeros(13), Q_sunPen=np.zeros(13), Q_ir=np.zeros(13),
                    Ww=0.5, Lr=np.zeros(13), Wr=0.5, Lc=np.zeros(13), H=np.zeros(13), C=np.zeros(13)):
        u_room = np.divide(self.k_2, self.R_inWall)*(Q_sunPen*Ww+Lr) + Q_sunPen*Wr+Lc+H-C
        u_wall = np.divide(self.k_2, self.R_room)*(Q_sunPen*Wr+Lr) + np.divide(self.k_1, self.R_ext)*(self.R_ext*T_ext + Q_sunExt + Q_ir)
        return u_room, u_wall
    
    def step(self, T_room, T_wall, u_room, u_wall, delta_t):
        T_dot_room, T_dot_wall = self.calculate_T_dot(T_room, T_wall, u_room, u_wall)
        T_room += delta_t * T_dot_room
        T_wall += delta_t * T_dot_wall
        return T_room, T_wall


if __name__ == "__main__":
    sim_asset = asset.get_asset()
    thermoPBM = thermoPBM(sim_asset)

    Q_sunPen = np.zeros(13)
    Q_sunExt = np.zeros(13)
    Q_ir = np.zeros(13)
    Lr = np.zeros(13)
    Lc = np.zeros(13)
    H = np.zeros(13)
    C = np.zeros(13)
    Ww = 0.5
    Wr = 0.5
    T_ext = 30

    delta_t = 60
    N = 1500
    
    T_room = np.full(13, 20.)
    T_wall = np.full(13, 20.)

    for i in range(N):
        u_room, u_wall = thermoPBM.calculate_U(T_ext, Q_sunExt, Q_sunPen, Q_ir, Ww, Lr, Wr, Lc, H, C)
        print(f'u_room: {u_room}')
        print(f'u_wall: {u_wall}')
        T_room, T_wall = thermoPBM.step(T_room, T_wall, u_room, u_wall, delta_t)
        print(f'T_room: {T_room}')
        print(f'T_wall: {T_wall}')