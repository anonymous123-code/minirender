import numpy as np
import matplotlib.pyplot as plt


def rotation_matrix(roll, nick, yaw):
    return np.matrix(
        [
            [np.cos(yaw) * np.cos(nick), np.sin(roll) * np.sin(nick) * np.cos(yaw) - np.sin(yaw) * np.cos(roll),
             np.sin(nick) * np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)],
            [np.sin(yaw) * np.cos(nick), np.sin(yaw) * np.sin(roll) * np.sin(nick) + np.cos(yaw) * np.cos(roll),
             np.sin(yaw) * np.cos(roll) * np.sin(nick) - np.sin(roll) * np.cos(yaw)],
            [-np.sin(nick), np.sin(roll) * np.cos(nick), np.cos(nick) * np.cos(roll)]
        ]
    )


class VertexGroup:
    def __init__(self, position, rotation_vector, position_vectors):
        self.position = position
        self.rotation_vector = rotation_vector
        self.position_vectors = position_vectors

    @property
    def position_relative_to_world(self) -> np.ndarray:
        return rotation_matrix(self.rotation_vector[0], self.rotation_vector[1],
                               self.rotation_vector[2]) * self.position_vectors + self.position


class Camera:
    def __init__(self, position, rotation_vector):
        self.position = position
        self.rotation_vector = rotation_vector

    def convert_to_camera_space(self, position_vectors):
        return rotation_matrix(-self.rotation_vector[0], -self.rotation_vector[1], -self.rotation_vector[2]) * (
                position_vectors - self.position
        )


if __name__ == "__main__":
    posVects = np.array(
        [
            [[-1],
             [-1],
             [-1]],

            [[1],
             [1],
             [-1]],

            [[1],
             [1],
             [1]],

            [[1],
             [-1],
             [-1]],

            [[1],
             [-1],
             [1]],

            [[-1],
             [1],
             [-1]],

            [[-1],
             [-1],
             [1]],

            [[-1],
             [1],
             [1]],
        ]
    )
    v = posVects.reshape(-1, 3).T

    group = VertexGroup(np.array([[1], [1], [1]]), np.array([0, 0, 0]), v)
    camera = Camera(np.array([[-1], [0], [0]]), np.array([0, 0, 0]))
    print(group.position_relative_to_world)
    print(camera.convert_to_camera_space(group.position_relative_to_world))


    def generate_frame():
        camera.position += np.array([[0.1], [0], [0]])
        group.rotation_vector += np.array([[0], [0], [0.1]])

        camera_space_coords = camera.convert_to_camera_space(group.position_relative_to_world)
        camera_space_coords[0] /= camera_space_coords[2]
        camera_space_coords[1] /= camera_space_coords[2]

        scaled_projected = (np.array(
            [1, 0, 0],
            [0, 1, 0]
        ) * camera_space_coords / 10)

        return scaled_projected[:, np.any(scaled_projected > 10, axis=0)]

    # def transform(t):
    #     return np.matrix([
    #         [np.cos(t), 0, -np.sin(t)],
    #         [0, 1, 0],
    #         [np.sin(t), 0, np.cos(t)],
    #     ])
    #
    #
    # projection = np.matrix([[1, 0, -.5],
    #                         [0, 1, -.5]])
    #
    # plt.ion()
    # plt.figure(0)
    # x = 0

    # def animate(i):
    #     x.append(np.random.rand(1) * 10)
    #     y.append(np.random.rand(1) * 10)
    #     sc.set_offsets(np.c_[x, y])
    #
    #
    # ani = matplotlib.animation.FuncAnimation(fig, animate,
    #                                          frames=2, interval=100, repeat=True)
    # while True:
    #     transformed = transform(x) * v
    #     print(transformed)
    #     projected = projection * transformed
    #     plt.scatter([projected[0]], [projected[1]])
    #     plt.axis(xmin=-5, xmax=5, ymin=-5, ymax=5)
    #     plt.draw()
    #     plt.pause(.2)
    #     plt.clf()
    #     x += 0.1
