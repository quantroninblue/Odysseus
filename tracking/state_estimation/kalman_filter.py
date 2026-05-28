"""
kalman_filter.py

Constant-velocity Kalman filter for 2D object tracking.

Purpose:
- Predict object motion
- Smooth noisy detections
- Stabilize temporal tracking
- Provide motion-aware state estimation

State Vector
------------
[x, y, vx, vy]

where:
x  -> center x
y  -> center y
vx -> velocity x
vy -> velocity y
"""

import numpy as np


class KalmanFilter2D:

    def __init__(
        self,
        dt=1.0,
        process_noise=1e-2,
        measurement_noise=1e-1
    ):

        self.dt = dt

        # --------------------------------------------------------
        # State vector
        # --------------------------------------------------------
        # [x, y, vx, vy]
        # --------------------------------------------------------

        self.x = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        # --------------------------------------------------------
        # State transition matrix
        # --------------------------------------------------------

        self.F = np.array([

            [1, 0, dt, 0],

            [0, 1, 0, dt],

            [0, 0, 1, 0],

            [0, 0, 0, 1]

        ], dtype=np.float64)

        # --------------------------------------------------------
        # Measurement matrix
        # --------------------------------------------------------
        # We only observe:
        # [x, y]
        # --------------------------------------------------------

        self.H = np.array([

            [1, 0, 0, 0],

            [0, 1, 0, 0]

        ], dtype=np.float64)

        # --------------------------------------------------------
        # State covariance
        # --------------------------------------------------------

        self.P = np.eye(
            4,
            dtype=np.float64
        ) * 100.0

        # --------------------------------------------------------
        # Process noise
        # --------------------------------------------------------

        self.Q = np.eye(
            4,
            dtype=np.float64
        ) * process_noise

        # --------------------------------------------------------
        # Measurement noise
        # --------------------------------------------------------

        self.R = np.eye(
            2,
            dtype=np.float64
        ) * measurement_noise

        # --------------------------------------------------------
        # Identity matrix
        # --------------------------------------------------------

        self.I = np.eye(
            4,
            dtype=np.float64
        )

    def initialize(
        self,
        x,
        y
    ):
        """
        Initialize filter state.
        """

        self.x = np.array([

            [x],
            [y],
            [0.0],
            [0.0]

        ], dtype=np.float64)

    def predict(self):
        """
        Predict next state.
        """

        # --------------------------------------------------------
        # State prediction
        # --------------------------------------------------------

        self.x = self.F @ self.x

        # --------------------------------------------------------
        # Covariance prediction
        # --------------------------------------------------------

        self.P = (

            self.F @ self.P @ self.F.T
            + self.Q
        )

        return self.get_state()

    def update(
        self,
        measured_x,
        measured_y
    ):
        """
        Update filter using measurement.
        """

        z = np.array([

            [measured_x],
            [measured_y]

        ], dtype=np.float64)

        # --------------------------------------------------------
        # Innovation
        # --------------------------------------------------------

        y = z - (self.H @ self.x)

        # --------------------------------------------------------
        # Innovation covariance
        # --------------------------------------------------------

        S = (

            self.H @ self.P @ self.H.T
            + self.R
        )

        # --------------------------------------------------------
        # Kalman gain
        # --------------------------------------------------------

        K = (

            self.P @ self.H.T @
            np.linalg.inv(S)
        )

        # --------------------------------------------------------
        # State update
        # --------------------------------------------------------

        self.x = self.x + (K @ y)

        # --------------------------------------------------------
        # Covariance update
        # --------------------------------------------------------

        self.P = (

            self.I - (K @ self.H)
        ) @ self.P

        return self.get_state()

    def get_state(self):
        """
        Return current estimated state.
        """

        return {

            "x": float(self.x[0, 0]),

            "y": float(self.x[1, 0]),

            "vx": float(self.x[2, 0]),

            "vy": float(self.x[3, 0])
        }


def main():

    print("\n=== Kalman Filter Validation ===\n")

    kf = KalmanFilter2D()

    kf.initialize(
        x=100,
        y=100
    )

    measurements = [

        (105, 102),
        (111, 105),
        (118, 109),
        (126, 114),
        (135, 120)
    ]

    for i, (mx, my) in enumerate(measurements):

        prediction = kf.predict()

        updated = kf.update(
            mx,
            my
        )

        print(
            f"\nStep {i}"
        )

        print(
            f"Measurement: ({mx}, {my})"
        )

        print(
            f"Prediction: "
            f"({prediction['x']:.2f}, "
            f"{prediction['y']:.2f})"
        )

        print(
            f"Updated: "
            f"({updated['x']:.2f}, "
            f"{updated['y']:.2f})"
        )

        print(
            f"Velocity: "
            f"({updated['vx']:.2f}, "
            f"{updated['vy']:.2f})"
        )


if __name__ == "__main__":
    main()