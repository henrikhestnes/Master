import copy
import torch
from torch import nn, optim


def pbm_temp_from_sensor(sensor_sample):
    batch_size = sensor_sample.shape[0]
    pbm_temp = torch.zeros((batch_size, 13))
    pbm_temp[:, 0] = sensor_sample[:, 0] #gfBedroom
    pbm_temp[:, 1] = sensor_sample[:, 1] #gfLivingroom
    pbm_temp[:, 2] = (sensor_sample[:, 9] + sensor_sample[:, 2])/2 #stairs
    pbm_temp[:, 3] = sensor_sample[:, 1] #gfBath
    pbm_temp[:, 4] = sensor_sample[:, 1] #gfStorage
    pbm_temp[:, 5] = sensor_sample[:, 3] #f1Guestroom
    pbm_temp[:, 6] = sensor_sample[:, 4] #f1Mainroom
    pbm_temp[:, 7] = sensor_sample[:, 4] #f1Sleep3
    pbm_temp[:, 8] = sensor_sample[:, 2] #f1Bath
    pbm_temp[:, 9] = sensor_sample[:, 2] #f1Storage
    pbm_temp[:, 10] = sensor_sample[:, 2] #f1Entrance
    pbm_temp[:, 11] = sensor_sample[:, 7] #f2Livingroom
    pbm_temp[:, 12] = sensor_sample[:, 8] #f2Office
    return pbm_temp


class FFCoSTA(nn.Module):
    def __init__(self, PBM, FFNN, temp_scaler=None):
        super(FFCoSTA, self).__init__()
        self.PBM = PBM
        self.DDM = FFNN
        self.temp_scaler = temp_scaler
    
    def forward(self, T_room, T_wall, T_out, door, timing, N, delta_t, num_preds):
        for i in range(num_preds):
            T_room_new = T_room.clone().requires_grad_(True).repeat(num_preds, 1, 1)
            T_wall_new = T_wall.clone().requires_grad_(True).repeat(num_preds, 1, 1)
            T_out_i = T_out[:, i].unsqueeze(1)
            door_i = door[:, i]
            timing_i = timing[:, i]
            for _ in range(N):
                T_room_hat, T_wall_hat = self.PBM(T_room_new[i], T_wall_new[i], T_out_i, delta_t)

                corrective_source_term = self.DDM(T_room_hat, T_wall_hat, T_out_i, door_i, timing_i)

                T_room_new[i], T_wall_new[i] = self.PBM(T_room_new[i], T_wall_new[i], T_out_i, delta_t, corrective_source_term)

        return T_room_new, T_wall_new
    
    def train(self, train_loader, val_loader, epochs, lr, l2_reg):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = lr)

        best_mae = float('inf')
        patience = 20
        i_since_last_update = 0

        delta_t = 60
        N = 15

        for epoch in range(epochs):
            train_mae = 0
            for i, (warmup_indoor, warmup_outdoor, indoor_temp, outdoor_temp, door, timing, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                T_room_warmup = pbm_temp_from_sensor(warmup_indoor[:, 0, :])
                T_wall_warmup = torch.zeros_like(T_room_warmup)
                for n in range(warmup_indoor.shape[1]):
                    for _ in range(N):
                        T_room_warmup, T_wall_warmup = self.PBM(T_room_warmup, T_wall_warmup, warmup_outdoor[:, n], delta_t)

                T_room = pbm_temp_from_sensor(indoor_temp).clone().requires_grad_(True)
                T_wall = T_wall_warmup

                num_preds = labels.shape[1]
                T_room_new, T_wall_new = self(T_room, T_wall, outdoor_temp, door, timing, N, delta_t, num_preds)

                label_compare_indices = [0, 1, 2, 3, 4, 7, 8]
                pbm_compare_indices = [0, 1, 10, 5, 6, 11, 12]
                
                labels = torch.swapaxes(labels, 0, 1)

                batch_mse = criterion(T_room_new[:, :, pbm_compare_indices], labels[:, :, label_compare_indices])

                reg_loss = 0
                for param in self.parameters():
                    reg_loss += torch.sum(torch.pow(param, 2))

                loss = batch_mse + l2_reg * reg_loss
                loss.backward()
                optimizer.step()

                # batch_mse = loss.item()
                # print(f'Batch: {i+1}, Batch Train MSE: {batch_mse}')
                train_mae += torch.mean(torch.abs(T_room_new[:, :, pbm_compare_indices] - labels[:, :, label_compare_indices]))
            train_mae /= (i+1)
            print(f'Epoch: {epoch+1}, Epoch Train MAE: {train_mae}')

            val_mae = 0
            for i, (warmup_indoor, warmup_outdoor, indoor_temp, outdoor_temp, door, timing, labels)  in enumerate(val_loader):
                T_room_warmup = pbm_temp_from_sensor(warmup_indoor[:, 0, :])
                T_wall_warmup = torch.zeros_like(T_room_warmup)
                
                for n in range(warmup_indoor.shape[1]):
                    for _ in range(N):
                        T_room_warmup, T_wall_warmup = self.PBM(T_room_warmup, T_wall_warmup, warmup_outdoor[:, n], delta_t)

                T_room = pbm_temp_from_sensor(indoor_temp).clone()
                T_wall = T_wall_warmup

                num_preds = labels.shape[1]
                T_room_new, T_wall_new = self(T_room, T_wall, outdoor_temp, door, timing, N, delta_t, num_preds)

                label_compare_indices = [0, 1, 2, 3, 4, 7, 8]
                pbm_compare_indices = [0, 1, 10, 5, 6, 11, 12]

                labels = torch.swapaxes(labels, 0, 1)

                val_mae  += torch.mean(torch.abs(T_room_new[:, :, pbm_compare_indices] - labels[:, :, label_compare_indices]))
            
            val_mae /= (i+1)
            print(f'Epoch: {epoch+1}, Val MAE: {val_mae}')

            #Early stopping
            if val_mae < best_mae:
                best_weights = copy.deepcopy(self.state_dict())
                i_since_last_update = 0
                best_mae = val_mae
            else:
                i_since_last_update += 1

            if i_since_last_update > patience:
                print(f"Stopping early with mae={best_mae}")
                break

        self.load_state_dict(best_weights)