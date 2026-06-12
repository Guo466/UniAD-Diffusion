# xorshift_python_normal.py - 与SDK C++完全一致的Xorshift随机数生成器
import struct
import math
import sys
from typing import List, Union, Optional
import numpy as np

# 64位掩码,用于模拟SDK C++的64位无符号整数运算
MASK64 = 0xFFFFFFFFFFFFFFFF

class XorShiftRandom:
    """与SDK C++版本完全一致的Xorshift随机数生成器"""
    
    def __init__(self, seed: int = 0x123456789ABCDEF):
        # 确保种子不为0,与SDK C++一致
        if seed == 0:
            seed = 1
        self.state = seed & MASK64
        self.cached_normal = 0.0
        self.has_cached_normal = False
    
    def seed(self, seed: int):
        """设置种子,与SDK C++版本完全一致"""
        if seed == 0:
            seed = 1
        self.state = seed & MASK64
        self.cached_normal = 0.0
        self.has_cached_normal = False
    
    def _next_uint64(self) -> int:
        """
        生成下一个64位随机整数,与SDK C++版本完全一致
        """
        x = self.state
        
        # 模拟SDK C++的64位无符号整数运算
        x ^= (x >> 12) & MASK64
        x ^= (x << 25) & MASK64
        x ^= (x >> 27) & MASK64
        
        self.state = x & MASK64
        
        # 乘法运算,使用Python的大整数,但结果会限制在64位
        result = (x * 0x2545F4914F6CDD1D) & MASK64
        
        return result
    
    def _next_uint32(self) -> int:
        """生成下一个32位随机整数,与SDK C++版本完全一致"""
        # 取高32位,与SDK C++的 static_cast<uint32_t>(next_uint64() >> 32) 一致
        return (self._next_uint64() >> 32) & 0xFFFFFFFF
    
    def random_double(self) -> float:
        """
        生成[0, 1)区间的double,53位精度,与SDK C++版本完全一致
        实现: (x >> 11) * (1.0 / 9007199254740992.0)
        """
        x = self._next_uint64()
        
        # 与SDK C++完全一致的操作：右移11位,然后除以2^53
        # 注意：在Python中,整数除法是精确的,这与SDK C++不同
        # 转换为浮点数,保证结果一致
        result = (x >> 11) * (1.0 / 9007199254740992.0)
        
        # 在SDK C++中,浮点数乘法可能导致精度问题,但算法上应该一致
        return result
    
    def random_float(self) -> float:
        """
        生成[0, 1)区间的float,24位精度,与SDK C++版本完全一致
        实现: (x >> 8) * (1.0 / 16777216.0)
        """
        x = self._next_uint32()
        
        # 与SDK C++完全一致的操作：右移8位,然后除以2^24
        result = (x >> 8) * (1.0 / 16777216.0)
        
        return result
    
    def random(self) -> float:
        """生成[0, 1)区间的double,与SDK C++版本完全一致"""
        return self.random_double()
    
    # ================ 高斯分布方法 ================
    
    def normal(self) -> float:
        """
        生成标准高斯分布 N(0, 1) 的随机数
        使用Box-Muller变换,与SDK C++版本完全一致
        """
        # 如果有缓存的值,直接返回
        if self.has_cached_normal:
            self.has_cached_normal = False
            return self.cached_normal
        
        # Box-Muller变换
        # 生成两个独立的均匀分布随机数
        # 注意：SDK C++中使用 do-while 循环避免 u1 <= 最小值
        u1 = 0.0
        u2 = 0.0
        
        # 使用Python的最小正浮点数,与SDK C++的std::numeric_limits<double>::min()对应
        # Python中为sys.float_info.min,约2.225e-308
        min_double = sys.float_info.min
        
        # 与SDK C++完全一致的do-while循环
        while True:
            u1 = self.random_double()
            if u1 > min_double:
                break
        
        u2 = self.random_double()
        
        # 极坐标形式的Box-Muller变换
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        
        # 一次生成两个独立的高斯分布随机数
        # 与SDK C++一致：缓存sin值,返回cos值
        self.cached_normal = r * math.sin(theta)
        self.has_cached_normal = True
        
        return r * math.cos(theta)
    
    def normal_with_params(self, mean: float = 0.0, stddev: float = 1.0) -> float:
        """
        生成高斯分布 N(mean, stddev^2) 的随机数
        与SDK C++的 normal(double mean, double stddev) 方法一致
        """
        if stddev <= 0:
            # 与SDK C++版本一致：标准差必须为正
            raise ValueError("Standard deviation must be positive")
        
        return mean + stddev * self.normal()
    
    def normal_float(self) -> float:
        """
        生成标准高斯分布的float
        与SDK C++的 normal_float() 方法一致
        """
        # 如果有缓存的值,直接返回
        if self.has_cached_normal:
            self.has_cached_normal = False
            return float(self.cached_normal)
        
        # Box-Muller变换
        u1 = 0.0
        u2 = 0.0
        
        min_double = sys.float_info.min
        
        while True:
            u1 = self.random_double()
            if u1 > min_double:
                break
        
        u2 = self.random_double()
        
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        
        self.cached_normal = r * math.sin(theta)
        self.has_cached_normal = True
        
        return float(r * math.cos(theta))
    
    def normal_float_with_params(self, mean: float = 0.0, stddev: float = 1.0) -> float:
        """生成高斯分布的float,与SDK C++一致"""
        if stddev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        return mean + stddev * self.normal_float()
    
    # ================ 矩阵生成方法 ================
    
    def random_vector_1d(self, size: int, 
                         min_val: float = 0.0, 
                         max_val: float = 1.0,
                         use_float: bool = False) -> List[float]:
        """生成1D随机向量,与SDK C++版本功能一致"""
        if size <= 0:
            raise ValueError("Vector size must be positive")
        
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val")
        
        vector = [0.0] * size
        range_val = max_val - min_val
        
        for i in range(size):
            if use_float:
                vector[i] = min_val + self.random_float() * range_val
            else:
                vector[i] = min_val + self.random_double() * range_val
        
        return vector
    
    def random_matrix_2d(self, rows: int, cols: int,
                         min_val: float = 0.0,
                         max_val: float = 1.0,
                         use_float: bool = False) -> List[List[float]]:
        """生成2D随机矩阵,与SDK C++版本功能一致"""
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive")
        
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val")
        
        matrix = [[0.0] * cols for _ in range(rows)]
        range_val = max_val - min_val
        
        for i in range(rows):
            for j in range(cols):
                if use_float:
                    matrix[i][j] = min_val + self.random_float() * range_val
                else:
                    matrix[i][j] = min_val + self.random_double() * range_val
        
        return matrix
    
    def normal_vector_1d(self, size: int,
                        mean: float = 0.0,
                        stddev: float = 1.0,
                        use_float: bool = False) -> List[float]:
        """生成1D高斯分布向量,与SDK C++版本功能一致"""
        if size <= 0:
            raise ValueError("Vector size must be positive")
        
        if stddev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        vector = [0.0] * size
        
        for i in range(size):
            if use_float:
                vector[i] = mean + stddev * self.normal_float()
            else:
                vector[i] = mean + stddev * self.normal()
        
        return vector
    
    def normal_matrix_2d(self, rows: int, cols: int,
                        mean: float = 0.0,
                        stddev: float = 1.0,
                        use_float: bool = False) -> List[List[float]]:
        """生成2D高斯分布矩阵,与SDK C++版本功能一致"""
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive")
        
        if stddev <= 0:
            raise ValueError("Standard deviation must be positive")
        
        matrix = [[0.0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                if use_float:
                    matrix[i][j] = mean + stddev * self.normal_float()
                else:
                    matrix[i][j] = mean + stddev * self.normal()
        
        return matrix
    
    # ================ NumPy转换方法 ================
    
    def to_numpy_random_1d(self, size: int,
                          min_val: float = 0.0,
                          max_val: float = 1.0,
                          dtype: type = np.float64) -> np.ndarray:
        """生成NumPy 1D随机数组"""
        use_float = (dtype == np.float32)
        vector = self.random_vector_1d(size, min_val, max_val, use_float)
        return np.array(vector, dtype=dtype)
    
    def to_numpy_random_2d(self, rows: int, cols: int,
                          min_val: float = 0.0,
                          max_val: float = 1.0,
                          dtype: type = np.float64) -> np.ndarray:
        """生成NumPy 2D随机数组"""
        use_float = (dtype == np.float32)
        matrix = self.random_matrix_2d(rows, cols, min_val, max_val, use_float)
        return np.array(matrix, dtype=dtype)
    
    def to_numpy_normal_1d(self, size: int,
                          mean: float = 0.0,
                          stddev: float = 1.0,
                          dtype: type = np.float64) -> np.ndarray:
        """生成NumPy 1D高斯分布数组"""
        use_float = (dtype == np.float32)
        vector = self.normal_vector_1d(size, mean, stddev, use_float)
        return np.array(vector, dtype=dtype)
    
    def to_numpy_normal_2d(self, rows: int, cols: int,
                          mean: float = 0.0,
                          stddev: float = 1.0,
                          dtype: type = np.float64) -> np.ndarray:
        """生成NumPy 2D高斯分布数组"""
        use_float = (dtype == np.float32)
        matrix = self.normal_matrix_2d(rows, cols, mean, stddev, use_float)
        return np.array(matrix, dtype=dtype)

def test_range_and_distribution():
    """测试随机数范围和分布"""
    print("\n测试随机数范围和分布...")
    
    rng = XorShiftRandom(seed=12345)
    
    # 测试random_double范围
    min_val = 1.0
    max_val = 0.0
    count = 10
    
    for i in range(count):
        val = rng.random_double()
        min_val = min(min_val, val)
        max_val = max(max_val, val)
    
    print(f"random_double() 测试 {count} 次:")
    print(f"  最小值: {min_val}")
    print(f"  最大值: {max_val}")
    print(f"  范围正确: {0.0 <= min_val and max_val < 1.0}")
    
    # 测试正态分布
    rng.seed(42)
    samples = [rng.normal() for _ in range(100)]
    
    mean_val = np.mean(samples)
    std_val = np.std(samples)
    
    print(f"\nnormal() 测试 {len(samples)} 个样本:")
    print(f"  均值: {mean_val:.6f} (期望: 0.0)")
    print(f"  标准差: {std_val:.6f} (期望: 1.0)")
    print(f"  分布接近标准正态: {abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.1}")


if __name__ == "__main__":
    # 运行测试
    test_range_and_distribution()