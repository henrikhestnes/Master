import numpy as np
import asset

# def skew_difference_matrix(T: np.ndarray) -> np.ndarray:
#     n = T.size
#     T_stack = np.tile(T, (1,n))
#     diff = T_stack - T_stack.T
#     return -diff

# def exchange_between_nodes(T: np.ndarray, A: np.ndarray) -> np.ndarray:
#     diff = skew_difference_matrix(T)
#     individual_exchange = np.multiply(A, diff)
#     net = np.sum(individual_exchange, axis=1, keepdims=True)
#     return net


# def get_rhs(T_rooms: np.ndarray,T_wall: np.ndarray,T_out: float, R_inv_internal: np.ndarray, R_inv_wall: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#         T_rooms: 1D array of air temperatures of the rooms
#         T_wall: 1D array of temperatures of the walls
#         T_out: temperature of the outside

#         R_inv_internal: 2D array of inverse resistances between rooms (symetric matrix with zero on diagonal)
#         R_inv_wall: 2D array of inverse resistances charaterizing the wall, with the following structure:
#             R_inv_ext: inverse resistance of the thin air layer at the exterior wall exterior surface
#             R_inv_out_wall: inverse resistance between the wall and the outside
#             R_inv_in_wall: inverse resistance between the wall and the inside
#             R_inv_room_wall: inverse resistance between the wall and the rooms
#             R_inv_outside: inverse resistance between the outside and the rooms in direct connection (e.g ventilation)

#         u: 2D array of source terms for the rooms, with the following structure:
#             u_out_wall: source term for heating the wall from the outside
#             u_in_wall: source term for heating the wall from the inside
#             u_direct: source term for heating directly in the rooms
#     """

#     internal_room_exchange = exchange_between_nodes(T_rooms, R_inv_internal)
    
#     R_inv_out_wall_outside = R_inv_wall[:,0].reshape(-1,1)
#     R_inv_out_wall = R_inv_wall[:,1].reshape(-1,1)
#     R_inv_in_wall = R_inv_wall[:,2].reshape(-1,1)
#     R_inv_room_wall = R_inv_wall[:,3].reshape(-1,1)
#     R_inv_outside = R_inv_wall[:,4].reshape(-1,1)

#     u_out_wall = u[:,0].reshape(-1,1)
#     u_in_wall = u[:,1].reshape(-1,1)
#     u_direct = u[:,2].reshape(-1,1)

#     directly_outside_exchange = R_inv_outside * (T_out - T_rooms)

#     R_inv_out_wall_sum = R_inv_out_wall + R_inv_out_wall_outside
#     R_inv_out_wall_sum[R_inv_out_wall_sum == 0] = 1

#     out_wall_to_wall_exchange = (R_inv_out_wall * u_out_wall + R_inv_out_wall*R_inv_out_wall_outside*(T_out - T_wall))/R_inv_out_wall_sum
#     print(R_inv_out_wall_sum)

#     in_wall_R_prod_T_diff = np.multiply(R_inv_in_wall*R_inv_room_wall, (T_rooms - T_wall))
#     in_wall_R_sum = R_inv_in_wall + R_inv_room_wall
#     in_wall_R_sum[in_wall_R_sum == 0] = 1

#     in_wall_to_wall_exchange = (R_inv_in_wall * u_in_wall + in_wall_R_prod_T_diff)/in_wall_R_sum

#     in_wall_to_room_exchange = (R_inv_room_wall * u_in_wall - in_wall_R_prod_T_diff)/in_wall_R_sum

#     rhs_rooms = internal_room_exchange + directly_outside_exchange + in_wall_to_room_exchange + u_direct 
#     rhs_wall = out_wall_to_wall_exchange + in_wall_to_wall_exchange
    
#     return rhs_rooms, rhs_wall

# def step(T_rooms: np.ndarray, T_wall: np.ndarray, T_out:float,R_inv_internal: np.ndarray, R_inv_wall: np.ndarray, u: np.ndarray,  C_inv_rooms: np.ndarray, C_inv_wall: np.ndarray, delta_t: float) -> tuple[np.ndarray, np.ndarray]:
#     rhs_rooms, rhs_wall = get_rhs(T_rooms, T_wall, T_out,R_inv_internal, R_inv_wall, u)
#     T_new_rooms = T_rooms + delta_t*C_inv_rooms*rhs_rooms
#     T_new_wall = T_wall + delta_t*C_inv_wall*rhs_wall
#     return T_new_rooms, T_new_wall


# if __name__ == "__main__":
#     #Asset values
#     sim_asset = asset.get_asset()
#     R_inv_internal = sim_asset.get_R_partWall_open_inv()
    
#     R_inv_wall = sim_asset.get_R_inv()

#     C_inv_rooms = sim_asset.get_C_open_inv()[0]
#     C_inv_walls = sim_asset.get_C_open_inv()[1]

#     #Initial values
#     T_rooms, T_wall, T_out = asset.get_initial_values()
#     T_rooms = T_rooms.reshape(-1,1)
#     T_wall = T_wall.reshape(-1,1)
#     T_out = T_out.reshape(-1,1)

#     u = np.array([
#         [0,0,0], 
#         [0,0,0], 
#         [0,0,0]
#         ])

#     delta_t = 1e-2
#     N = 5000
#     for i in range(N):
#         T_rooms, T_wall = step(T_rooms, T_wall, T_out, R_inv_internal, R_inv_wall, u,  C_inv_rooms, C_inv_walls, delta_t)
#         print(f"T_ROOMS{T_rooms}")
#         print(f"T_WALL{T_wall}")

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

        self.k_1 = np.divide(np.matmul(R_internal[0], R_internal[1]), R_internal[0] + R_internal[1])
        self.k_2 = np.divide(np.matmul(R_internal[2], R_internal[3]), R_internal[2] + R_internal[3])

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
    
    def calculate_U(self, T_ext, Q_sunExt, Q_sunPen, Q_ir, Ww, Lr, Wr, Lc, H, C):
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

    delta_t = 1e-3
    N = 10000
    
    T_room = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    T_wall = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    for i in range(N):
        u_room, u_wall = thermoPBM.calculate_U(T_ext, Q_sunExt, Q_sunPen, Q_ir, Ww, Lr, Wr, Lc, H, C)
        print(f'u_room: {u_room}')
        print(f'u_wall: {u_wall}')
        T_room, T_wall = thermoPBM.step(T_room, T_wall, u_room, u_wall, delta_t)
        print(f'T_room: {T_room}')
        print(f'T_wall: {T_wall}')