import numpy as np

from transforms_reference import tf_to_matrix


class DummyTranslation:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class DummyRotation:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class DummyTransform:
    def __init__(self):
        self.translation = DummyTranslation(
            1.0,
            2.0,
            3.0
        )

        self.rotation = DummyRotation(
            0.0,
            0.0,
            0.0,
            1.0
        )


class DummyTF:
    def __init__(self):
        self.transform = DummyTransform()


def main():

    tf_msg = DummyTF()

    T = tf_to_matrix(tf_msg)

    print("\nTransform Matrix:\n")
    print(T)

    print("\nShape:")
    print(T.shape)

    print("\nDeterminant of rotation block:")
    print(np.linalg.det(T[:3, :3]))

    print("\nTranslation:")
    print(T[:3, 3])

    print("\n--- Point Transform Test ---")

    point_camera = np.array([
        1.0,
        1.0,
        1.0,
        1.0
    ])

    point_world = T @ point_camera

    print("\nCamera-frame point:")
    print(point_camera)

    print("\nWorld-frame point:")
    print(point_world)


if __name__ == "__main__":
    main()