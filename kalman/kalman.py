import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import rts_smoother

from utils.uwb_model import UWBModel
from utils.movement_model import MovementModel
from aremnet.model import ARemNet

class EKFLocalization:
    def __init__(self, uwb_model: UWBModel, movement_model: MovementModel, mitigation_model: ARemNet, dt=0.1, process_var=0.1, meas_var=0.5):
        self.uwb_model = uwb_model
        self.movement_model = movement_model
        self.mitigation_model = mitigation_model

        self.dt = dt
        self.beacons = uwb_model.origins 

        self.ekf = EKF(dim_x=4, dim_z=1)

        self.ekf.x = np.array([3.3, 1, 0, 0])

        self.ekf.F = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

        self.ekf.R = np.array([[meas_var]])

        self.ekf.Q = process_var

        self.ekf.P *= 1

        self.true_positions = np.zeros((uwb_model.t.shape[0], 2))

        for t_i, t in enumerate(uwb_model.t):
            self.true_positions[t_i] = self.movement_model.get_xy(t)

    def motion_model(self, x):
        F = self.ekf.F
        return np.dot(F, x)

    def measurement_model(self, x, beacon):
        return np.array([np.sqrt((x[0] - beacon[0]) ** 2 + (x[1] - beacon[1]) ** 2)])

    def jacobian_H(self, x, beacon):
        px, py, _, _ = x
        bx, by, _ = beacon
        d = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
        
        if d == 0:
            return np.zeros((1, 4))

        return np.array([[(px - bx) / d, (py - by) / d, 0, 0]])

    def apply_rts(self, xs_post, Ps_post, Fs, Qs):
        xs_post = np.array(xs_post)
        Ps_post = np.array(Ps_post)
        Fs = np.array(Fs)
        Qs = np.array(Qs)

        xs, _, _, _ = rts_smoother(
            xs_post, Ps_post, Fs, Qs,
        )

        return xs[:, :2]

    def infer(self, model, data):
        model.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(0)
            data = data.to(model.device)

            output = model(data)
            return output.squeeze().cpu().numpy()
        
    def run_ekf(self, mitigate=True):
        xs_post = []
        Ps_post = []
        Fs = []
        Qs = []

        results = []
        nis_list = []
        nees_list = []

        start_idx = 0
        end_idx = self.uwb_model.t.shape[0]

        for t_i in tqdm(range(start_idx, end_idx)):
            z = np.array([self.uwb_model.get_dist(t_i)])
            beacon = np.array(self.uwb_model.get_origin(t_i))

            if mitigate:
                cir = np.array([self.uwb_model.get_cir(t_i)])[0]
                cir = np.array([np.array([x.real, x.imag]).T for x in cir])

                fp_idx = 8
                prior_fp = 5
                after_fp = 128
                cir = cir[fp_idx - prior_fp: fp_idx + after_fp]
                error = self.infer(self.mitigation_model, cir) / 100

                z -= error
            
            dt = self.uwb_model.t[t_i] - self.uwb_model.t[t_i - 1] if t_i > 0 else self.dt
            self.ekf.F = np.array([[1, 0, dt, 0],
                                   [0, 1, 0, dt],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
            
            Fs.append(self.ekf.F.copy())
            Qs.append(self.ekf.Q.copy())

            self.ekf.predict()

            H = self.jacobian_H(self.ekf.x, beacon)

            x_prior = self.ekf.x.copy()
            P_prior = self.ekf.P.copy()

            z_pred = self.measurement_model(x_prior, beacon)

            y = z - z_pred
            S = H @ P_prior @ H.T + self.ekf.R

            nis = y.T @ np.linalg.inv(S) @ y
            nis_list.append(nis.item())

            self.ekf.update(z, HJacobian=lambda x: H, Hx=lambda x: self.measurement_model(x, beacon))

            xs_post.append(self.ekf.x.copy())
            Ps_post.append(self.ekf.P.copy())

            results.append(self.ekf.x[:2])

        trajectory = self.apply_rts(xs_post, Ps_post, Fs, Qs)
        true = self.true_positions

        _, path = fastdtw(results, true, dist=euclidean)

        nees_list = []
        for i_est, i_true in path:
            e = results[i_est] - true[i_true]
            P_pos = Ps_post[i_est][:2, :2]
            nees = e.T @ np.linalg.inv(P_pos) @ e
            nees_list.append(nees.item())

        return {
            "trajectory": np.array(trajectory),
            "true_positions": self.true_positions,
            "nis": np.array(nis_list),
            "nees": np.array(nees_list),
        }
