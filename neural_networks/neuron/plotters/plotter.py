import matplotlib.pyplot as plt


def plot(points, weights):
    x1, x2 = zip(*points)
    plt.plot(x1, x2, 'o')
    y_0 = max(x2)
    x_1 = max(x1)
    y_1 = -(weights[0] + weights[1] * x_1) / weights[2]
    x_0 = -(weights[0] + weights[2] * y_0) / weights[1]
    plt.plot([x_0, x_1], [y_0, y_1])
    plt.show()


if __name__ == "__main__":
    plot(points=[(0, 0), (1, 0.1), (1.15, 0.9), (-0.1, 1.2)], weights=[.7, 0.5, 0.5])
    # plot(points=[(-1, -1), (1, -1.1), (1.15, 0.9), (-1.1, 1.2)], weights=[.7, 0.5, 0.5])
    plot(points=[(0, 0), (1, 0.1), (1.15, 0.9), (-0.1, 1.2)], weights=[-0.2777777, 0.5, 0.5])
