# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
# plt.rcParams['font.size'] = 14  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


datasets = [("../data/n10-1682-X-small.npy", "../data/n10-1682-y-small.npy"),
            ("../data/n10-3127-X-small.npy", "../data/n10-3127-y-small.npy"),
            ("../data/n10-6797-X-small.npy", "../data/n10-6797-y-small.npy"),
            ("../data/n20-6174-X-small.npy", "../data/n20-6174-y-small.npy")]
dataset_n = 4


def four_cdfs():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    flat_axes = [ax for ax_row in axes for ax in ax_row]

    for i in range(dataset_n):
        X = np.load(datasets[i][0])
        y = np.load(datasets[i][1]).astype(np.float64)
        y /= np.max(y)

        flat_axes[i].plot(X, y)
        flat_axes[i].set_title("数据集D%i" % (i+1))

    axes[1][0].set_xlabel("搜索键")
    axes[1][1].set_xlabel("搜索键")
    axes[0][0].set_ylabel("累积分布函数")
    axes[1][0].set_ylabel("累积分布函数")

    fig.tight_layout()
    fig.savefig("four-cdfs.pdf")
    plt.close(fig)


def dispatch():
    X = np.load("../data/dispatch.npy")
    y = np.linspace(0, 1, len(X))
    mean = 2000

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(X, y, label="实际分配情况")
    ax.vlines(mean, 0, 1, label="理想均匀分配情况")

    ax.set_xlabel("递归索引模型最后级机器学习模型训练集数据量")
    ax.set_ylabel("对数正态分布下递归索引模型\n数据分配的累积分布函数")
    ax.set_xlim(left=-100, right=20000)
    ax.legend(loc=4)

    fig.tight_layout()
    fig.savefig("dispatch.pdf")
    plt.close(fig)


def osm_cdfs():
    X = np.load("../data/osm-sizes-X-small.npy")
    y = np.load("../data/osm-sizes-y-small.npy").astype(np.float64)
    y /= np.max(y)

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(X, y)

    ax.set_xlabel("搜索键")
    ax.set_ylabel("OSM数据集的累积分布函数")

    fig.tight_layout()
    fig.savefig("osm-cdf.pdf")
    plt.close(fig)


def stretching():
    hot_data_start = 0.4
    hot_data_ratio = 0.3
    hot_query_ratio = 0.7

    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

    X = np.load("../data/n10-1682-X-small.npy")
    y = np.load("../data/n10-1682-y-small.npy").astype(np.float64)
    y /= np.max(y)

    axes[0].set_xlabel("搜索键")
    axes[0].set_ylabel("累积分布函数")
    axes[0].plot(X[:int(len(X) * hot_data_start)],
                 y[:int(len(X) * hot_data_start)],
                 "C0")
    axes[0].plot(X[int(len(X) * hot_data_start):
                   int(len(X) * (hot_data_start + hot_data_ratio))],
                 y[int(len(X) * hot_data_start):
                   int(len(X) * (hot_data_start + hot_data_ratio))],
                 "C1")
    axes[0].plot(X[int(len(X) * (hot_data_start + hot_data_ratio)):],
                 y[int(len(X) * (hot_data_start + hot_data_ratio)):],
                 "C0")

    accum_freq_pivot_1 = (1 - hot_query_ratio) * \
        hot_data_start / (1 - hot_data_ratio)
    accum_freq_pivot_2 = accum_freq_pivot_1 + hot_query_ratio

    accum_freq_1 = np.linspace(0, accum_freq_pivot_1,
                               int(len(X) * hot_data_start))
    accum_freq_2 = np.linspace(accum_freq_pivot_1, accum_freq_pivot_2,
                               int(len(X) * (hot_data_start + hot_data_ratio)) - int(len(X) * hot_data_start))
    accum_freq_3 = np.linspace(accum_freq_pivot_2, 1,
                               len(X) - int(len(X) * (hot_data_start + hot_data_ratio)))

    axes[1].plot(X[:int(len(X) * hot_data_start)],
                 accum_freq_1, "C0")
    axes[1].plot(X[int(len(X) * hot_data_start):
                   int(len(X) * (hot_data_start + hot_data_ratio))],
                 accum_freq_2, "C1")
    axes[1].plot(X[int(len(X) * (hot_data_start + hot_data_ratio)):],
                 accum_freq_3, "C0")

    axes[1].set_xlabel("搜索键")
    axes[1].set_ylabel("“拉伸”后的累积分布函数")

    fig.tight_layout()
    fig.savefig("stretching.pdf")
    plt.close(fig)


def stretching_result():
    original_lat = np.asarray((270.84, 260.47, 288.27))
    stretching_lat = np.asarray((236.70, 222.96, 156.98))
    original = 1000 / original_lat
    stretching = 1000 / stretching_lat

    ind = np.arange(len(original))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.bar(ind - width/2, original, width, label='原始方法')
    ax.bar(ind + width/2, stretching, width, label='数据拉伸')

    ax.set_ylabel('学习索引结构吞吐量')
    ax.set_xticks(ind)
    ax.set_xticklabels(('Skewed 1', 'Skewed 2', 'Skewed 3'))
    ax.legend()

    fig.tight_layout()
    fig.savefig("stretching-result.pdf")
    plt.close(fig)


def analyzer():
    fig, ax = plt.subplots(figsize=(4, 3))

    X = np.load("../data/osm-sizes-X-small.npy")
    y = np.load("../data/osm-sizes-y-small.npy").astype(np.float64)
    y /= np.max(y)
    K = 50

    sample_X = []
    sample_y = []

    max_X = np.max(X)
    min_X = np.min(X)

    i = 0
    last_X = X[0]
    last_y = y[0]
    for point_i in range(len(X)):
        desired_X = i / K * (max_X - min_X) + min_X

        if X[point_i] < desired_X:
            last_X = X[point_i]
            last_y = y[point_i]
            continue

        if y[point_i] - last_y == 0:
            estimated_y = 0
        else:
            estimated_y = y[point_i] - (X[point_i] - desired_X) / \
                (X[point_i] - last_X) * (y[point_i] - last_y)
        sample_X.append(desired_X)
        sample_y.append(estimated_y)

        last_X = X[point_i]
        last_y = y[point_i]
        i += 1

    sample_X = np.asarray(sample_X)
    sample_y = np.asarray(sample_y)

    ax.plot(X, y, label="原始训练集")
    ax.plot(sample_X, sample_y, ".", label="分析器抽样样本")

    ax.set_xlabel("搜索键")
    ax.set_ylabel("“拉伸”后的累积分布函数")
    ax.legend()

    fig.tight_layout()
    fig.savefig("analyzer.pdf")
    plt.close(fig)


def counselor():
    best = (2.24, 2.11, 2.13, 2.28)
    counselor = (2.194793946, 1.876994334, 2.032794003, 2.26594023)

    ind = np.arange(len(best))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.bar(ind - width/2, best, width, label='网格搜索产生的最优架构')
    ax.bar(ind + width/2, counselor, width, label='咨询器产生的最优架构')

    ax.set_ylabel('学习索引结构吞吐量')
    ax.set_ylim(bottom=0, top=3)
    ax.set_xticks(ind)
    ax.set_xticklabels(('D1', 'D2', 'D3', 'D4'))
    ax.legend()

    fig.tight_layout()
    fig.savefig("counselor.pdf")
    plt.close(fig)


if __name__ == "__main__":
    four_cdfs()
    dispatch()
    osm_cdfs()
    stretching()
    stretching_result()
    analyzer()
    counselor()
