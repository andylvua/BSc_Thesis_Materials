import numpy as np

class MovementModel:
    def __init__(self, df, x_name="x", y_name="y", z_name="z"):
        self.tss = np.array(df["t"])
        self.xss = np.array(df[x_name], dtype=np.float64)
        self.yss = np.array(df[y_name], dtype=np.float64)

        dts = np.ediff1d(self.tss, to_begin=1)
        self.vx = np.ediff1d(self.xss, to_begin=0.0) / (dts + 1e-40)
        self.vy = np.ediff1d(self.yss, to_begin=0.0) / (dts + 1e-40)

    def get_xy(self, t):
        return np.array([np.interp(t, self.tss, self.xss), np.interp(t, self.tss, self.yss)])
