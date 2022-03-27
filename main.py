import numpy as np
import matplotlib.pyplot as plt


def rotMatrix(roll, nick, yaw):
    return np.matrix(
        [
            [np.cos(roll) * np.cos(nick), np.sin(yaw) * np.sin(nick) * np.cos(roll) - np.sin(yaw) * np.cos(roll),
             np.sin(nick) * np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)],
            [np.sin(roll) * np.cos(nick), np.sin(yaw) * np.sin(roll) * np.cos(nick) + np.cos(yaw) * np.cos(roll),
             np.sin(yaw) * np.cos(roll) * np.cos(nick) - np.sin(roll) * np.cos(yaw)],
            [-np.sin(nick), np.sin(roll) * np.cos(nick), np.cos(nick) * np.cos(roll)]
        ]
    )


class VertexGroup:
    def __init__(self, pos, rotVec, posVects):
        self.pos = pos
        self.rotVec = rotVec
        self.posVects = posVects

    def getPosRelativeToWorld(self):
        return rotMatrix(self.rotVec[0], self.rotVec[1], self.rotVec[2]) * self.posVects + self.pos


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

    group = VertexGroup(np.array([[1], [1], [1]]), np.array([np.pi*2, 0, 0]), v)
    group.getPosRelativeToWorld()


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
