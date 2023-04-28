import numpy as np

class Zone:
    def __init__(self, zone_name: str, R: list[float], C: list[float]):
        """
            zone_name: name of zone:)
            R: 1D array of inverse resistance in the zone in the following order:
                R_ext: Resistance of the thin air layer at the exterior wall exterior surface
                R_outWall: Resistance of the outer part of the exterior wall
                R_inWall: Resistance of the inner part of the exterior wall
                R_room: Resistance of the thin air layer at the exterior wall interior surface
            C: 1D array of inverse capasitance in the zone in the following order:
                C_wall: Capacitance of the heavy wall material in the room
                C_room: Capacitance of the air and furniture in the room
        """
        self.name = zone_name
        self.R_values = R
        self.C_values = C
    

    def get_name(self):
        return self.name

    def get_R(self):
        return self.R_values
    
    def get_C(self):
        return self.C_values


class Asset:
    def __init__(self, zones: list[Zone], connections: list[dict]):
        """
            zones: List of all zones in asset
            connections: Dictionary of all connections in asset
        """
        self.zones = {zone.get_name(): zone for zone in zones}

        self.R_ext = np.zeros(len(zones))
        self.R_outWall = np.zeros(len(zones))
        self.R_inWall = np.zeros(len(zones))
        self.R_room = np.zeros(len(zones))

        self.C_room = np.zeros((len(zones)))
        self.C_wall = np.zeros((len(zones)))

        for i, zone in enumerate(zones):
            zone_R = zone.get_R()
            self.R_ext[i] = zone_R[0]
            self.R_outWall[i] = zone_R[1]
            self.R_inWall[i] = zone_R[2]
            self.R_room[i] = zone_R[3]

            zone_C = zone.get_C()
            self.C_room[i] = zone_C[0]
            self.C_wall[i] = zone_C[0]
        
        self.R_partWall = np.zeros((len(zones), len(zones)))
        self.C_partWall = np.zeros((len(zones), len(zones)))
        for connection in connections:
            room_1_index = list(self.zones.keys()).index(connection['rooms'][0])
            room_2_index = list(self.zones.keys()).index(connection['rooms'][1])

            self.R_partWall[room_1_index, room_2_index] = connection['R']
            self.R_partWall[room_2_index, room_1_index] = connection['R']
            
            self.C_partWall[room_1_index, room_2_index] = connection['C']
            self.C_partWall[room_2_index, room_1_index] = connection['C']

            
        for row in range(len(zones)):
            for col in range(len(zones)):
                self.C_room[row] += 0.25*self.C_partWall[row, col] if self.C_partWall[row, col] != np.inf else 0
                self.C_wall[row] += 0.25*self.C_partWall[row, col] if self.C_partWall[row, col] != np.inf else 0


    def get_R(self):
        return np.array([self.R_ext, self.R_outWall, self.R_inWall, self.R_room])

    def get_C(self):
        return np.array([self.C_room, self.C_wall])    
    
    def get_R_partWall(self):
        return self.R_partWall
    

gfBedroom = Zone("gfBedroom",       [0.1, 1, 1, 1], [1, 1])
gfLivingroom = Zone("gfLivingroom", [0.1, 1, 1, 1], [1, 1])
stairs = Zone("stairs",             [0.1, 1, 1, 1], [1, 1])
gfBath = Zone("gfBath",             [0.1, 1, 1, 1], [1, 1])
gfStorage = Zone("gfStorage",       [0.1, 1, 1, 1], [1, 1])
f1Guestroom = Zone("f1Guestroom",   [0.1, 1, 1, 1], [1, 1])
f1Mainroom = Zone("f1Mainroom",     [0.1, 1, 1, 1], [1, 1])
f1Sleep3 = Zone("f1Sleep3",         [0.1, 1, 1, 1], [1, 1])
f1Bath = Zone("f1Bath",             [0.1, 1, 1, 1], [1, 1])
f1Storage = Zone("f1Storage",       [0.1, 1, 1, 1], [1, 1])
f1Entrance = Zone("f1Entrance",     [0.1, 1, 1, 1], [1, 1])
f2Livingroom = Zone("f2Livingroom", [0.1, 1, 1, 1], [1, 1])
f2Office = Zone("f2Office",         [0.1, 1, 1, 1], [1, 1])
zones = [gfBedroom, gfLivingroom, stairs, gfBath, gfStorage, f1Guestroom, f1Mainroom, f1Sleep3, f1Bath, f1Storage, f1Entrance, f2Livingroom, f2Office]

connections =  [{"rooms": ["gfBedroom", "gfLivingroom"],    "R": 1, "C": 1}, 
                {"rooms": ["gfBedroom", "stairs"],          "R": 1, "C": 1},
                {"rooms": ["gfBedroom", "f1Guestroom"],     "R": 1, "C": 1},
                {"rooms": ["gfLivingroom", "gfBath"],       "R": 1, "C": 1},
                {"rooms": ["gfLivingroom", "stairs"],       "R": 1, "C": 1},
                {"rooms": ["gfLivingroom", "f1Entrance"],   "R": 1, "C": 1},
                {"rooms": ["gfLivingroom", "f1Mainroom"],   "R": 1, "C": 1},
                {"rooms": ["gfBath", "gfStorage"],          "R": 1, "C": 1},
                {"rooms": ["gfBath", "f1Sleep3"],           "R": 1, "C": 1},
                {"rooms": ["gfStorage", "stairs"],          "R": 1, "C": 1},
                {"rooms": ["gfStorage", "f1Bath"],          "R": 1, "C": 1},
                {"rooms": ["f1Guestroom", "f1Mainroom"],    "R": 1, "C": 1},
                {"rooms": ["f1Guestroom", "f1Entrance"],    "R": 1, "C": 1},
                {"rooms": ["f1Guestroom", "stairs"],        "R": 1, "C": 1},
                {"rooms": ["f1Guestroom", "f2Livingroom"],  "R": 1, "C": 1},
                {"rooms": ["f1Mainroom", "f1Sleep3"],       "R": 1, "C": 1},
                {"rooms": ["f1Mainroom", "f1Entrance"],     "R": 1, "C": 1},
                {"rooms": ["f1Mainroom", "f2Livingroom"],   "R": 1, "C": 1},
                {"rooms": ["f1Sleep3", "f1Entrance"],       "R": 1, "C": 1},
                {"rooms": ["f1Sleep3", "f2Livingroom"],     "R": 1, "C": 1},
                {"rooms": ["f1Entrance", "f1Bath"],         "R": 1, "C": 1},
                {"rooms": ["f1Entrance", "stairs"],         "R": 1, "C": 1},
                {"rooms": ["f1Bath", "f1Storage"],          "R": 1, "C": 1},
                {"rooms": ["f1Bath", "stairs"],             "R": 1, "C": 1},
                {"rooms": ["f1Bath", "f2Office"],           "R": 1, "C": 1},
                {"rooms": ["f2Livingroom", "f2Office"],     "R": 1, "C": 1},
                {"rooms": ["f2Livingroom", "stairs"],       "R": 1, "C": 1},
                {"rooms": ["f2Office", "stairs"],           "R": 1, "C": 1}]

asset = Asset(zones, connections)

out_temperature = 25
initial_zone_temperature = [18, 18, 18]
initial_wall_temperature = [20, 20, 20]

def get_asset():
    return asset

def get_initial_values():
    return np.array(initial_zone_temperature), np.array(initial_wall_temperature), np.array(out_temperature)

