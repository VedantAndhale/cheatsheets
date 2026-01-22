# 🔢 NumPy Cheatsheet

**Author:** @VedantAndhale  
**Purpose:** Comprehensive yet scannable NumPy notes with parameter details, method references, and common traps.

---

## Table of Contents
1. [NumPy Basics](#1-numpy-basics)
2. [Why NumPy? (Memory & Speed)](#2-why-numpy-memory--speed)
3. [Creating Arrays](#3-creating-arrays)
4. [ndarray Attributes](#4-ndarray-attributes)
5. [Data Types](#5-data-types)
6. [Random Functions](#6-random-functions)
7. [Indexing & Slicing](#7-indexing--slicing)
8. [Fancy Indexing & Boolean Masking](#8-fancy-indexing--boolean-masking)
9. [Descriptive Statistics](#9-descriptive-statistics)
10. [Math & Matrix Operations](#10-math--matrix-operations)
11. [Broadcasting](#11-broadcasting)
12. [Stacking & Splitting](#12-stacking--splitting)
13. [Sorting & Logic](#13-sorting--logic)
14. [Linear Algebra](#14-linear-algebra)
15. [File I/O](#15-file-io)
16. [Common Traps](#16-common-traps)
17. [Quick Reference](#17-quick-reference)

---

## 1. NumPy Basics

**NumPy** = **Num**erical **Py**thon  
- Created in **2005** by **Travis Oliphant**
- Foundation for scientific computing in Python
- Core object: **ndarray** (N-dimensional array)

### Import Convention
```python
import numpy as np  # Standard alias
```

### Check Version
```python
np.__version__  # e.g., '1.24.0'
```

---

## 2. Why NumPy? (Memory & Speed)

### ⚡ Speed Comparison
NumPy is **10-100x faster** than pure Python lists for numerical operations.

```python
import numpy as np
import time

# NumPy (FAST) - Vectorized operation
large_array = np.arange(10_000_000)
start = time.time()
result = large_array * 2
numpy_time = time.time() - start

# Python list (SLOW) - Loop required
large_list = list(range(10_000_000))
start = time.time()
result = [x * 2 for x in large_list]
python_time = time.time() - start

print(f"NumPy was {python_time / numpy_time:.1f}x faster!")
```

### 💾 Memory Comparison
NumPy arrays use **less memory** than Python lists.

```python
from pympler import asizeof

array = np.array([1, 2, 3, 4, 5])
lst = [1, 2, 3, 4, 5]

print('Array size:', asizeof.asizeof(array))  # ~160 bytes
print('List size:', asizeof.asizeof(lst))     # ~344 bytes
```

### Why NumPy is Fast
| Factor | Explanation |
|--------|-------------|
| **Contiguous memory** | Data stored in continuous block |
| **C-based algorithms** | Avoids Python interpreter overhead |
| **Vectorized operations** | No Python loops needed |
| **Locality of reference** | Cache-friendly memory access |

---

## 3. Creating Arrays

### Manual Creation with `np.array()`

```python
# 0-D (Scalar)
scalar = np.array(42)                    # shape: ()

# 1-D (Vector)
vector = np.array([1, 2, 3])             # shape: (3,)

# 2-D (Matrix)
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])           # shape: (2, 3)

# 3-D (Tensor)
tensor = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])    # shape: (2, 2, 2)
```

### Array Creation Functions

| Function | Syntax | Description |
|----------|--------|-------------|
| `np.array()` | `np.array(object, dtype=None)` | Create array from list/tuple |
| `np.zeros()` | `np.zeros(shape, dtype=float)` | Array filled with zeros |
| `np.ones()` | `np.ones(shape, dtype=float)` | Array filled with ones |
| `np.empty()` | `np.empty(shape)` | Uninitialized array (⚠️ garbage values) |
| `np.full()` | `np.full(shape, fill_value)` | Array filled with specified value |
| `np.eye()` | `np.eye(N, M=None)` | 2D array with ones on diagonal |
| `np.identity()` | `np.identity(n)` | Square identity matrix |
| `np.arange()` | `np.arange(start, stop, step)` | Values with given step (like `range`) |
| `np.linspace()` | `np.linspace(start, stop, num)` | Evenly spaced values over interval |

```python
np.zeros((2, 3))           # [[0. 0. 0.]
                           #  [0. 0. 0.]]

np.ones((2, 3), dtype=int) # [[1 1 1]
                           #  [1 1 1]]

np.full((2, 3), 7)         # [[7 7 7]
                           #  [7 7 7]]

np.eye(3)                  # [[1. 0. 0.]
                           #  [0. 1. 0.]
                           #  [0. 0. 1.]]

np.arange(0, 10, 2)        # [0 2 4 6 8]

np.linspace(0, 1, 5)       # [0.   0.25 0.5  0.75 1.  ]
```

### `*_like` Functions (Match Shape)

| Function | Syntax | Description |
|----------|--------|-------------|
| `np.zeros_like()` | `np.zeros_like(arr)` | Zeros array with same shape |
| `np.ones_like()` | `np.ones_like(arr)` | Ones array with same shape |
| `np.empty_like()` | `np.empty_like(arr)` | Empty array with same shape |
| `np.full_like()` | `np.full_like(arr, fill_value)` | Filled array with same shape |

```python
base = np.array([[1, 2], [3, 4]])
np.zeros_like(base)  # [[0 0]
                     #  [0 0]]
```

### Reshaping Arrays

```python
arr = np.arange(1, 10)     # [1 2 3 4 5 6 7 8 9]
arr.reshape(3, 3)          # [[1 2 3]
                           #  [4 5 6]
                           #  [7 8 9]]

# Use -1 to auto-calculate dimension
arr.reshape(3, -1)         # Same as (3, 3)
arr.reshape(-1, 1)         # Column vector (9, 1)
```

---

## 4. ndarray Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `.shape` | Tuple of dimensions | `(3, 3)` for 3×3 matrix |
| `.size` | Total number of elements | `9` for 3×3 matrix |
| `.ndim` | Number of dimensions | `2` for matrix |
| `.dtype` | Data type of elements | `int64`, `float64` |
| `.itemsize` | Bytes per element | `8` for int64 |
| `.strides` | Bytes to step in each dimension | `(24, 8)` for 3×3 int64 |
| `.flags` | Memory layout info | C_CONTIGUOUS, F_CONTIGUOUS |

```python
arr = np.arange(1, 10).reshape(3, 3)

arr.shape      # (3, 3)
arr.size       # 9
arr.ndim       # 2
arr.dtype      # dtype('int64')
arr.itemsize   # 8
arr.strides    # (24, 8) - 24 bytes to next row, 8 to next col
```

### Understanding Strides ⚡

Strides tell NumPy how many bytes to jump to reach the next element.

```python
arr = np.arange(1, 10).reshape(3, 3)  # int64 = 8 bytes each
# arr.strides = (24, 8)
# → Move 24 bytes (3 × 8) to go to next row
# → Move 8 bytes (1 × 8) to go to next column
```

---

## 5. Data Types

### NumPy Data Types Table

| Type | Code | Description | Size |
|------|------|-------------|------|
| `int8` | `i1` | Signed 8-bit integer | 1 byte |
| `int16` | `i2` | Signed 16-bit integer | 2 bytes |
| `int32` | `i4` | Signed 32-bit integer | 4 bytes |
| `int64` | `i8` | Signed 64-bit integer | 8 bytes |
| `uint8` | `u1` | Unsigned 8-bit integer | 1 byte |
| `uint32` | `u4` | Unsigned 32-bit integer | 4 bytes |
| `float16` | `f2` | Half-precision float | 2 bytes |
| `float32` | `f4` | Single-precision float | 4 bytes |
| `float64` | `f8` | Double-precision float (default) | 8 bytes |
| `bool` | `?` | Boolean (True/False) | 1 byte |
| `str_` | `U` | Unicode string | Variable |

**Defaults:** `int64` for integers, `float64` for floats

### Specifying dtype

```python
np.array([1, 2, 3], dtype='float32')   # Explicit type
np.zeros((2, 2), dtype=np.int32)       # Using np.int32
np.ones((3,), dtype='i4')              # Using code
```

### Type Conversion with `astype()`

```python
arr = np.array([1.5, 2.7, 3.9])
arr.astype(int)     # [1 2 3] - truncates decimals
arr.astype('str')   # ['1.5' '2.7' '3.9']
```

### Type Promotion Hierarchy ⚠️

When mixing types, NumPy **promotes** to the highest type:

$$\text{bool} \rightarrow \text{int} \rightarrow \text{float} \rightarrow \text{str}$$

```python
np.array([1, 2.5, True])       # float64: [1.  2.5 1. ]
np.array([1, 2, "hello"])      # str: ['1' '2' 'hello']
np.array([True, 42, 3.14])     # float64: [1.   42.   3.14]
```

### Memory Optimization Tips ⚡

```python
# Use smaller dtypes when precision allows
big = np.arange(1000000, dtype='int64')    # ~8 MB
small = np.arange(1000000, dtype='int16')  # ~2 MB (4x smaller!)

# Check memory usage
arr.nbytes  # Total bytes used
```

---

## 6. Random Functions

### Random Number Generation

| Function | Syntax | Description | Range |
|----------|--------|-------------|-------|
| `np.random.rand()` | `rand(d0, d1, ...)` | Uniform distribution | [0, 1) |
| `np.random.uniform()` | `uniform(low, high, size)` | Uniform distribution | [low, high) |
| `np.random.randn()` | `randn(d0, d1, ...)` | Standard normal | (-∞, +∞) |
| `np.random.normal()` | `normal(loc, scale, size)` | Normal distribution | (-∞, +∞) |
| `np.random.randint()` | `randint(low, high, size)` | Random integers | [low, high) |
| `np.random.choice()` | `choice(arr, size, replace)` | Random selection | From array |
| `np.random.shuffle()` | `shuffle(arr)` | Shuffle in-place | Modifies array |
| `np.random.seed()` | `seed(n)` | Set random seed | For reproducibility |

### Uniform Distribution

```python
# [0, 1) - default range
np.random.rand(3)           # 1D: [0.374, 0.951, 0.732]
np.random.rand(2, 3)        # 2D: shape (2, 3)

# [low, high) - custom range
np.random.uniform(10, 20, size=5)        # 1D
np.random.uniform(0, 100, size=(2, 3))   # 2D
```

### Normal (Gaussian) Distribution

```python
# Standard normal: mean=0, std=1
np.random.randn(5)                       # 5 random values

# Custom normal: specify mean (loc) and std (scale)
np.random.normal(loc=100, scale=15, size=1000)  # IQ-like distribution
```

### Random Integers

```python
np.random.randint(1, 7)              # Single die roll [1, 7)
np.random.randint(0, 100, size=5)    # 5 random ints [0, 100)
np.random.randint(1, 10, size=(3, 3)) # 3×3 matrix [1, 10)
```

### Random Sampling

```python
elements = np.array(['a', 'b', 'c', 'd'])

# With replacement (can repeat)
np.random.choice(elements, size=3, replace=True)   # ['b', 'b', 'd']

# Without replacement (unique only)
np.random.choice(elements, size=3, replace=False)  # ['c', 'a', 'd']

# Shuffle array in-place
np.random.shuffle(elements)  # Modifies original!
```

### Reproducible Random Numbers ⚠️

```python
np.random.seed(42)
print(np.random.rand(3))  # [0.3745..., 0.9507..., 0.7319...]

np.random.seed(42)        # Reset seed
print(np.random.rand(3))  # Same output! [0.3745..., 0.9507..., 0.7319...]
```

**Use `seed()` for:**
- Reproducible experiments
- Debugging random code
- Consistent test results

### Modern Generator API (NumPy 1.17+) ✅ Recommended

The new Generator API is **preferred** over legacy `np.random` functions:

```python
# Create a Generator with a seed
rng = np.random.default_rng(seed=42)

# Uniform [0, 1)
rng.random(size=(3, 3))

# Integers [low, high)
rng.integers(low=1, high=10, size=5)

# Normal distribution
rng.normal(loc=0, scale=1, size=100)

# Random choice
rng.choice(['a', 'b', 'c'], size=3, replace=False)

# Shuffle (returns copy, doesn't modify original!)
rng.permutation(arr)  # ✅ Returns shuffled copy
rng.shuffle(arr)      # Modifies in-place
```

**Why use Generator over legacy?**
| Feature | Legacy (`np.random`) | Generator (`default_rng`) |
|---------|---------------------|---------------------------|
| Thread safety | ❌ Global state | ✅ Independent |
| Reproducibility | ⚠️ Fragile | ✅ Robust |
| Performance | Good | Better |
| Future-proof | Deprecated path | Recommended |

---

## 7. Indexing & Slicing

### 1D Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element
arr[0]      # 10 (first)
arr[-1]     # 50 (last)
arr[2]      # 30 (third)

# Slicing: arr[start:stop:step]
arr[1:4]    # [20 30 40]
arr[:3]     # [10 20 30] (first 3)
arr[2:]     # [30 40 50] (from index 2)
arr[::2]    # [10 30 50] (every 2nd)
arr[::-1]   # [50 40 30 20 10] (reverse)
```

### 2D Indexing

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Single element (2 ways)
matrix[1][2]    # 6 - chained indexing
matrix[1, 2]    # 6 - tuple indexing (preferred ✅)

# Access rows
matrix[0]       # [1 2 3] (first row)
matrix[0:2]     # [[1 2 3], [4 5 6]] (first 2 rows)
matrix[-1]      # [7 8 9] (last row)

# Access columns
matrix[:, 0]    # [1 4 7] (first column)
matrix[:, -1]   # [3 6 9] (last column)
matrix[:, 0:2]  # [[1 2], [4 5], [7 8]] (first 2 columns)

# Submatrix
matrix[0:2, 0:2]  # [[1 2], [4 5]] (top-left 2×2)
matrix[1:, 1:]    # [[5 6], [8 9]] (bottom-right)
```

### 2D Access Patterns Summary

| Pattern | Syntax | Result |
|---------|--------|--------|
| Single element | `matrix[row, col]` | Scalar |
| Entire row | `matrix[row]` or `matrix[row, :]` | 1D array |
| Entire column | `matrix[:, col]` | 1D array |
| Row range | `matrix[r1:r2]` | 2D submatrix |
| Column range | `matrix[:, c1:c2]` | 2D submatrix |
| Submatrix | `matrix[r1:r2, c1:c2]` | 2D submatrix |

---

## 8. Fancy Indexing & Boolean Masking

### Fancy Indexing (Index with Arrays)

Select multiple elements using an array of indices:

```python
arr = np.array([10, 20, 30, 40, 50])

# Select multiple indices
arr[[0, 2, 4]]          # [10 30 50]
arr[[4, 2, 0]]          # [50 30 10] (can reorder!)
arr[[0, 0, 1, 1]]       # [10 10 20 20] (can repeat!)
```

### 2D Fancy Indexing

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select specific rows
matrix[[0, 2]]              # [[1 2 3], [7 8 9]]

# Select specific columns
matrix[:, [0, 2]]           # [[1 3], [4 6], [7 9]]

# Select diagonal elements
matrix[[0, 1, 2], [0, 1, 2]]  # [1 5 9]

# Select corners
matrix[[0, 0, 2, 2], [0, 2, 0, 2]]  # [1 3 7 9]
```

### Boolean Masking

Filter arrays using boolean conditions:

```python
arr = np.array([10, 15, 20, 25, 30])

# Create boolean mask
mask = arr > 20           # [False False False True True]

# Apply mask
arr[mask]                 # [25 30]

# Direct filtering (preferred)
arr[arr > 20]             # [25 30]
arr[arr % 2 == 0]         # [10 20 30] (even numbers)

# Multiple conditions (use & for AND, | for OR)
arr[(arr > 15) & (arr < 30)]  # [20 25]
arr[(arr < 15) | (arr > 25)]  # [10 30]
```

### Custom Boolean Mask

```python
data = np.array([42, 47, -1, 89, 0])
keep = np.array([True, False, True, False, True])

data[keep]  # [42 -1 0]
```

### Indexing Methods Comparison ⚠️

| Method | Syntax | Returns | Memory |
|--------|--------|---------|--------|
| Single index | `arr[i]` | Element | N/A |
| Slice | `arr[start:stop:step]` | **View** | Shared ⚠️ |
| Fancy indexing | `arr[[i, j, k]]` | **Copy** | Independent |
| Boolean mask | `arr[mask]` | **Copy** | Independent |

**View vs Copy:**
```python
arr = np.array([1, 2, 3, 4, 5])

# Slice = VIEW (shares memory!)
view = arr[1:4]
view[0] = 99
print(arr)  # [1 99 3 4 5] ← Original changed!

# Fancy indexing = COPY (independent)
copy = arr[[1, 2, 3]]
copy[0] = 100
print(arr)  # [1 99 3 4 5] ← Original unchanged
```

---

## 9. Descriptive Statistics

### Aggregation Functions

| Function | Description | NaN Safe Version |
|----------|-------------|------------------|
| `np.sum(arr)` | Sum of elements | `np.nansum()` |
| `np.mean(arr)` | Arithmetic mean | `np.nanmean()` |
| `np.median(arr)` | Median value | `np.nanmedian()` |
| `np.min(arr)`, `np.max(arr)` | Minimum/Maximum value | `np.nanmin()`, `np.nanmax()` |
| `np.std(arr)`, `np.var(arr)` | Standard Deviation / Variance | `np.nanstd()`, `np.nanvar()` |
| `np.argmin(arr)`, `np.argmax(arr)` | Index of min/max value | `np.nanargmin()`, `np.nanargmax()` |

### The `axis` Parameter ⚠️
- **`axis=0`**: "Vertical" — collapse rows (calculate column stats).
- **`axis=1`**: "Horizontal" — collapse columns (calculate row stats).

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

np.sum(arr, axis=0)  # [5, 7, 9]  (Sum of columns)
np.sum(arr, axis=1)  # [6, 15]    (Sum of rows)
```

### Handling Missing Values (NaN)
Standard functions return `nan` if ANY value is `nan`. Use `nan*` functions.

```python
a = np.array([1, 2, np.nan])
np.mean(a)      # nan
np.nanmean(a)   # 1.5 (ignores nan)
```

---

## 10. Math & Matrix Operations

### Arithmetic (Element-wise)
Works with scalars or arrays of same shape.
`+`, `-`, `*`, `/`, `**` (power), `%` (mod)

```python
x = np.array([1, 2]); y = np.array([3, 4])
x + y      # [4, 6]
x * y      # [3, 8]
np.sqrt(x) # [1., 1.41]
```

### Matrix Multiplication (Dot Product)
| Method | Syntax | Note |
|--------|--------|------|
| **Operator** | `A @ B` | **Preferred** (Python 3.5+) |
| Function | `np.matmul(A, B)` | Same as `@` |
| Dot | `np.dot(A, B)` | Classic (handles scalars differently) |

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

result = A @ B  # [[19, 22], [43, 50]]
```

---

## 11. Broadcasting

**Rule:** Arrays can be operated on if their dimensions are **equal** or **one of them is 1**.

1. **Scalar to Array:**
   ```python
   np.array([1, 2, 3]) + 5   # [6, 7, 8]
   ```

2. **Vector to Matrix:**
   ```python
   matrix = np.ones((3, 3))  # 3x3
   row = np.array([0, 1, 2]) # 1x3 (broadcasts down)
   matrix + row
   # [[1., 2., 3.],
   #  [1., 2., 3.],
   #  [1., 2., 3.]]
   ```

3. **Incompatible:** `(3, 3) + (3,)` works, but `(3, 3) + (2,)` fails.

---

## 12. Stacking & Splitting

### Combining Arrays

| Function | Description | Result Shape (2D ex) |
|----------|-------------|----------------------|
| `np.vstack((a, b))` | Stack vertically (rows) | `(N+M, C)` |
| `np.hstack((a, b))` | Stack horizontally (cols) | `(R, N+M)` |
| `np.concatenate((a, b), axis=...)` | General join | Depends on axis |

```python
a = np.array([1, 2])
b = np.array([3, 4])

np.vstack((a, b))  # [[1, 2],
                   #  [3, 4]]
np.hstack((a, b))  # [1, 2, 3, 4]
```

### Splitting Arrays
- `np.split(arr, N)`: Split into N equal parts (error if not equal).
- `np.array_split(arr, N)`: Split into N parts (allows unequal).

---

## 13. Sorting & Logic

### Sorting
```python
arr = np.array([30, 10, 20])

np.sort(arr)          # [10, 20, 30] (returns sorted copy)
print(arr)            # [30, 10, 20] (original unchanged)

arr.sort()            # Sort in-place (modifies arr)
print(arr)            # [10, 20, 30]

# argsort - indices that would sort the array
arr = np.array([30, 10, 20])
np.argsort(arr)       # [1, 2, 0] → arr[[1,2,0]] = [10, 20, 30]
```

### Conditional Logic (`np.where`) ⚡

**Syntax:** `np.where(condition, value_if_true, value_if_false)`

```python
arr = np.array([10, 25, 30, 5])

# Vectorized if-else
np.where(arr < 20, 0, arr)  # [0, 25, 30, 0]

# Replace specific values
np.where(arr == 30, -1, arr) # [10, 25, -1, 5]
```

### Unique Values (`np.unique`)

```python
arr = np.array([3, 1, 2, 1, 3, 3, 2])

np.unique(arr)                      # [1 2 3] (sorted unique)
np.unique(arr, return_counts=True)  # ([1 2 3], [2 2 3])
np.unique(arr, return_index=True)   # ([1 2 3], [1 2 0]) first occurrence
```

### Clipping Values (`np.clip`)

```python
arr = np.array([1, 5, 10, 15, 20])

np.clip(arr, 5, 15)     # [5 5 10 15 15] - clamp to range
np.clip(arr, None, 10)  # [1 5 10 10 10] - cap maximum only
np.clip(arr, 5, None)   # [5 5 10 15 20] - set minimum only
```

### Adding Dimensions (`np.newaxis`)

```python
arr = np.array([1, 2, 3])  # shape: (3,)

# Add dimension
arr[:, np.newaxis]  # shape: (3, 1) - column vector
arr[np.newaxis, :]  # shape: (1, 3) - row vector

# Equivalent to:
arr.reshape(-1, 1)  # (3, 1)
arr.reshape(1, -1)  # (1, 3)
np.expand_dims(arr, axis=1)  # (3, 1)
```

---

## 14. Linear Algebra

### Basic Operations

```python
import numpy.linalg as la

A = np.array([[1, 2], [3, 4]])

# Transpose
A.T                    # [[1 3], [2 4]]

# Determinant
la.det(A)              # -2.0

# Inverse
la.inv(A)              # [[-2.   1. ]
                       #  [ 1.5 -0.5]]

# Matrix rank
la.matrix_rank(A)      # 2

# Trace (sum of diagonal)
np.trace(A)            # 5 (1 + 4)
```

### Eigenvalues & Eigenvectors

```python
eigenvalues, eigenvectors = la.eig(A)
# eigenvalues: [-0.37, 5.37]
# eigenvectors: columns are eigenvectors
```

### Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = la.solve(A, b)  # [2. 3.] → 3*2 + 1*3 = 9, 1*2 + 2*3 = 8
```

### Norms

```python
v = np.array([3, 4])

la.norm(v)        # 5.0 (Euclidean/L2 norm)
la.norm(v, ord=1) # 7.0 (L1 norm: |3| + |4|)
la.norm(v, ord=np.inf)  # 4.0 (max absolute value)
```

### SVD (Singular Value Decomposition)

```python
U, S, Vt = la.svd(A)
# A = U @ np.diag(S) @ Vt
```

---

## 15. File I/O

### Binary Files (.npy, .npz)

```python
arr = np.array([1, 2, 3, 4, 5])

# Save single array
np.save('array.npy', arr)
loaded = np.load('array.npy')

# Save multiple arrays (compressed)
np.savez('arrays.npz', x=arr, y=arr*2)
data = np.load('arrays.npz')
data['x']  # [1 2 3 4 5]
data['y']  # [2 4 6 8 10]

# Compressed version (slower but smaller)
np.savez_compressed('arrays_compressed.npz', x=arr, y=arr*2)
```

### Text Files (.txt, .csv)

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save to text file
np.savetxt('array.txt', arr, delimiter=',', fmt='%d')
np.savetxt('array.csv', arr, delimiter=',', header='a,b,c')

# Load from text file
loaded = np.loadtxt('array.txt', delimiter=',')
loaded = np.genfromtxt('array.csv', delimiter=',', skip_header=1)
```

### Memory-Mapped Files (Large Arrays)

```python
# Create memory-mapped file (doesn't load into RAM)
large_arr = np.memmap('large.dat', dtype='float32', mode='w+', shape=(10000, 10000))
large_arr[0, :] = np.arange(10000)  # Write to disk
del large_arr  # Flush to disk

# Read back
mmap = np.memmap('large.dat', dtype='float32', mode='r', shape=(10000, 10000))
print(mmap[0, :10])  # Read without loading entire file
```

---

## 16. Common Traps

### ⚠️ Trap 1: View vs Copy

**Slicing creates a VIEW** that shares memory with original:

```python
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]   # VIEW
view[0] = 99
print(original)  # [1 99 3 4 5] ← Modified!

# Solution: Explicit copy
safe_copy = original[1:4].copy()
safe_copy[0] = 100
print(original)  # [1 99 3 4 5] ← Unchanged
```

### ⚠️ Trap 2: `randint()` High is Exclusive

```python
np.random.randint(1, 10)  # Returns 1-9, NEVER 10!

# If you want 1-10 inclusive:
np.random.randint(1, 11)  # [1, 11) = 1 to 10
```

### ⚠️ Trap 3: `normal()` Scale Must Be Positive

```python
# ❌ This will raise an error
np.random.normal(loc=0, scale=-1, size=5)

# ✅ Scale (std deviation) must be positive
np.random.normal(loc=0, scale=1, size=5)
```

### ⚠️ Trap 4: `empty()` Returns Garbage

```python
arr = np.empty((2, 3))
# Contains whatever was in memory - NOT zeros!
# [[4.67e-310 0.00e+000 2.12e-314]
#  [2.14e-314 2.14e-314 2.14e-314]]

# Use np.zeros() if you need initialized values
```

### ⚠️ Trap 5: Shape Tuple Quirks

```python
# These are ALL different shapes:
np.array([1, 2, 3]).shape         # (3,)   - 1D array
np.array([[1, 2, 3]]).shape       # (1, 3) - row vector
np.array([[1], [2], [3]]).shape   # (3, 1) - column vector

# (3,) vs (3, 1) can cause broadcasting issues!
a = np.array([1, 2, 3])           # shape (3,)
b = np.array([[1], [2], [3]])     # shape (3, 1)
# a + b broadcasts to shape (3, 3)!
```

### ⚠️ Trap 6: In-place Operations

```python
arr = np.array([1, 2, 3])

# These DON'T modify arr:
arr * 2        # Returns new array
arr + 10       # Returns new array

# These DO modify arr:
arr *= 2       # In-place multiplication
arr += 10      # In-place addition
np.random.shuffle(arr)  # In-place shuffle
```

### ⚠️ Trap 7: Boolean Operators

```python
arr = np.array([1, 2, 3, 4, 5])

# ❌ Python's and/or don't work!
# arr[(arr > 2) and (arr < 5)]  # ValueError!

# ✅ Use & for AND, | for OR, ~ for NOT
arr[(arr > 2) & (arr < 5)]   # [3 4]
arr[(arr < 2) | (arr > 4)]   # [1 5]
arr[~(arr > 3)]              # [1 2 3]
```

---

## 17. Quick Reference

### Import
```python
import numpy as np
```

### Array Creation
```python
np.array([1, 2, 3])           # From list
np.zeros((3, 3))              # All zeros
np.ones((2, 4))               # All ones
np.full((2, 2), 7)            # All 7s
np.eye(3)                     # Identity matrix
np.arange(0, 10, 2)           # [0 2 4 6 8]
np.linspace(0, 1, 5)          # 5 evenly spaced
np.empty((2, 2))              # Uninitialized
```

### Attributes
```python
arr.shape      # Dimensions tuple
arr.size       # Total elements
arr.ndim       # Number of dimensions
arr.dtype      # Data type
arr.itemsize   # Bytes per element
```

### Reshaping
```python
arr.reshape(3, 3)    # New shape
arr.reshape(-1, 1)   # Auto-calculate rows
arr.flatten()        # To 1D (copy)
arr.ravel()          # To 1D (view)
```

### Random
```python
np.random.seed(42)              # Reproducibility
np.random.rand(3, 3)            # Uniform [0, 1)
np.random.randn(3, 3)           # Normal (0, 1)
np.random.randint(1, 100, (3,)) # Integers [1, 100)
np.random.choice(arr, 3)        # Random selection
np.random.shuffle(arr)          # In-place shuffle
```

### Indexing
```python
arr[0]              # First element
arr[-1]             # Last element
arr[1:4]            # Slice
arr[::2]            # Every 2nd
arr[::-1]           # Reverse

matrix[0, 1]        # Element at row 0, col 1
matrix[0]           # First row
matrix[:, 0]        # First column
matrix[0:2, 0:2]    # Submatrix
```

### Boolean Filtering
```python
arr[arr > 5]                    # Greater than 5
arr[(arr > 2) & (arr < 8)]      # Between 2 and 8
arr[arr % 2 == 0]               # Even numbers
```

### Type Conversion
```python
arr.astype(float)    # To float
arr.astype('int32')  # To int32
arr.astype(str)      # To string
```

### Useful Checks
```python
np.may_share_memory(a, b)  # Check if shared
arr.flags                  # Memory layout
arr.nbytes                 # Total bytes
```

### Vectorize Custom Functions
```python
# Apply Python function element-wise
def my_func(x):
    return x ** 2 if x > 0 else 0

vectorized = np.vectorize(my_func)
vectorized(np.array([-1, 0, 1, 2]))  # [0 0 1 4]
```

### Coordinate Grids (`meshgrid`)
```python
x = np.array([1, 2, 3])
y = np.array([4, 5])

X, Y = np.meshgrid(x, y)
# X: [[1 2 3], [1 2 3]]
# Y: [[4 4 4], [5 5 5]]

# Useful for: plotting, coordinate calculations
Z = X**2 + Y**2  # Apply formula to grid
```

---

**📝 Notes:**
- Always use `np.random.seed()` for reproducible results
- Prefer `matrix[i, j]` over `matrix[i][j]` for 2D access
- Remember: slicing = view, fancy indexing = copy
- Use `arr.copy()` when you need an independent copy

---

