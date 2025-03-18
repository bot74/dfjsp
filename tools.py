import numpy as np
import random

def split_by_slices(data, slices, fill_value= 0):
    """
    将数据按照 slices 的分组方式排列，并填充为固定大小的数组。

    参数:
        data (list): 输入数据，例如 [1, 2, 3, 4, 5, 6]。
        slices (list): 分组方式，例如 [1, 3, 2]。
        fill_value (any): 填充值，默认为 '*'。

    返回:
        np.ndarray: 分组后的数组，填充为固定大小。
    """
    # 计算目标数组的行数和列数
    num_rows = len(slices)  # 行数等于 slices 的长度
    num_cols = max(slices)  # 列数等于 slices 中的最大值

    # 初始化结果数组
    result = np.full((num_rows, num_cols), fill_value, dtype=object)

    # 填充数据
    start = 0
    for i, size in enumerate(slices):
        end = start + size
        result[i, :size] = data[start:end]
        start = end

    return result

from itertools import permutations

def generate_job_permutations(operations):
    """
    生成所有可能的工序排列（去除空组合），并与工件名字结合为字典。

    参数:
        operations (list): 工序列表，例如 [0, 1, 2]。

    返回:
        dict: 键为工件名字，值为工序排列的字典。
    """
    # 生成工件名字（a, b, c, ...）
    job_names = [chr(ord('a') + i) for i in range(len(operations))]
    
    # 生成所有可能的工序排列（去除空组合）
    all_permutations = []
    for r in range(1, len(operations) + 1):  # 从 1 开始，去除空组合
        permutations_r = list(permutations(operations, r))
        all_permutations.extend(permutations_r)
    
    # 将工序排列与工件名字结合为字典
    job_dict = {}
    for i, perm in enumerate(all_permutations):
        job_name = i  # 工件名字为 1,2,3, ...
        job_dict[job_name] = list(perm)
    
    return job_dict

def generate_length_list(m, wc):
    """
    生成表示每个子列表长度的列表。

    参数:
    - m: 整数 表示总范围 0 到 m-1。
    - wc: 整数，表示子列表的数量。

    返回:
    - length_list: 列表，表示每个子列表的长度。
    """
    # 生成 0 到 m-1 的整数列表
    full_range = list(range(m))

    # 如果 wc 大于 m，无法生成 wc 个子列表
    if wc > m:
        raise ValueError("wc cannot be greater than m.")

    # 随机生成 wc-1 个分割点
    split_points = sorted(random.sample(range(1, m), wc - 1))

    # 根据分割点切片
    wc_list = []
    start = 0
    for end in split_points:
        wc_list.append(full_range[start:end])
        start = end
    wc_list.append(full_range[start:])  # 添加最后一部分

    # 生成表示每个子列表长度的列表
    length_list = [len(sublist) for sublist in wc_list]

    return length_list

def get_group_index(slices, number):
    """
    根据切片 slices 和整数 number 返回 number 对应的组索引。
    
    参数:
        slices (list): 切片列表，例如 [1, 3, 2]。
        number (int): 需要查找的整数，例如 0-5。
    
    返回:
        int: 组索引。
    """
    start = 0
    for group_index, group_size in enumerate(slices):
        end = start + group_size
        if start <= number < end:
            return group_index
        start = end
    raise ValueError(f"Number {number} is out of range for slices {slices}")