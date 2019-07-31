import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

def get_positional_encoding(max_seq_len,embed_dim):
    # 初始化一个positional encoding
    # max_seq_len:字嵌入维度
    # embed_dim:最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    print(positional_encoding.shape) # 100,16
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数，隔2列取一个值
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数，隔2列取一个值

    # 归一化，用位置嵌入的每一行除以它的模长
    # denominator = np.sqrt(np.sum(positional_encoding**2, axis=1, keepdims=True))
    # positional_encoding = positional_encoding / (denominator + 1e-8)
    return positional_encoding

if __name__ == "__main__":
    positional_encoding = get_positional_encoding(max_seq_len=100,embed_dim=16)
    plt.figure(figsize=(10, 10))
    sns.heatmap(positional_encoding)
    plt.title("sin function")
    plt.xlabel("hidden dimension")
    plt.ylabel("sequence dimension")
    plt.savefig('./output_img/sns_heatmap.png')

    plt.figure(figsize=(8,4))
    plt.plot(positional_encoding[1:,0],label='dimension 0')
    plt.plot(positional_encoding[1:,1],label='dimension 1')
    plt.plot(positional_encoding[1:,2],label='dimension 2')
    plt.plot(positional_encoding[1:,3],label='dimension 3')
    plt.legend()
    plt.xlabel('Sequence Length')
    plt.ylabel('Period of Positional Encoding')
    plt.savefig('./output_img/period_of_positional_encoding.png')