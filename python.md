# 🐍 Python Cheatsheet 

**Author:** @VedantAndhale  
**Purpose:** Comprehensive yet scannable notes with parameter details and method references.

---

## Table of Contents
1. [Python Basics](#1-python-basics)
2. [Variables & Data Types](#2-variables--data-types)
3. [Operators](#3-operators)
4. [Strings](#4-strings)
5. [Lists](#5-lists)
6. [Tuples](#6-tuples)
7. [Sets](#7-sets)
8. [Dictionaries](#8-dictionaries)
9. [Conditional Statements](#9-conditional-statements)
10. [Loops](#10-loops)
11. [Comprehensions](#11-comprehensions)
12. [Functions](#12-functions)
13. [Iterators & Generators](#13-iterators--generators)
14. [Decorators](#14-decorators)
15. [OOP Basics](#15-oop-basics)
16. [Inheritance & Method Overriding](#16-inheritance--method-overriding)
17. [Encapsulation](#17-encapsulation)
18. [Abstraction](#18-abstraction)
19. [Polymorphism](#19-polymorphism)
20. [Instance/Class/Static Methods](#20-instanceclassstatic-methods)
21. [Dunder/Magic Methods](#21-dundermagic-methods)
22. [File Handling](#22-file-handling)
23. [Exception Handling](#23-exception-handling)
24. [Shallow vs Deep Copy](#24-shallow-vs-deep-copy)
25. [Type Hints](#25-type-hints)
26. [Dataclasses](#26-dataclasses)
27. [Match Statement](#27-match-statement)
28. [Walrus Operator](#28-walrus-operator-)
29. [Context Managers](#29-context-managers-with-statement)
30. [Async/Await](#30-asyncawait)
31. [Common Traps](#31-common-traps)
32. [Quick Reference](#32-quick-reference)

---
 
## 1. Python Basics
- **Interpreted:** No compilation needed, runs directly
- **Dynamically typed:** Variable types determined at runtime
- **Multi-paradigm:** OOP, functional, procedural
- **Run:** `python filename.py` or use REPL: `python` / `ipython`

### `__name__ == "__main__"` (⚠️ Common)
```python
if __name__ == "__main__":
    main()  # Only runs when script is executed directly
```
- When **run directly:** `__name__` = `"__main__"` → code inside runs
- When **imported:** `__name__` = `"module_name"` → code inside skipped

---

## 2. Variables & Data Types
**Common types:** `int`, `float`, `str`, `bool`, `None`, `list`, `tuple`, `set`, `dict`  
**Naming:** snake_case; can't start with digit; avoid keywords like `if`, `for`, `class`

```python
name = "Vedant"  # str
age = 21         # int
price = 9.99     # float
is_active = True # bool
```

**Type conversion:** `int("42")`, `float("3.14")`, `str(100)`, `bool(1)`

### Mutable vs Immutable (⚠️ Very Important)
| Mutable (Can Change) | Immutable (Cannot Change) |
|---------------------|---------------------------|
| `list` | `int`, `float`, `bool` |
| `dict` | `str` |
| `set` | `tuple` |
| | `frozenset` |
| | `None` |

**Why it matters:**
```python
# Immutable - creates new object
s = "hello"
s[0] = "H"  # ❌ TypeError!

# Mutable - modifies same object
l = [1, 2, 3]
l[0] = 99   # ✅ Works! l = [99, 2, 3]
```

---

## 3. Operators
**Arithmetic:** `+ - * / // % **`  
- `//` = floor division: `7 // 2 = 3`  
- `%` = modulo (remainder): `7 % 2 = 1`  
- `**` = power: `2 ** 3 = 8`

**Comparison (Relational):** `== != > < >= <=` → returns `True/False`  
**Logical:** `and`, `or`, `not`  
**Membership:** `in`, `not in` → check if item exists  
**Identity:** `is`, `is not` → check if same object

```python
5 > 3 and 2 < 4  # True
'a' in 'apple'   # True
x is None        # True if x is None
```
### Short-Circuiting ⚡ (Important!)
**Python stops evaluating as soon as result is determined**

#### `and` Short-Circuit
- Returns **first falsy value** OR **last value** if all truthy
- Stops at first `False`

```python
False and expensive_function()  # Returns False, function NOT called
True and True and False and print("Never runs")  # Stops at False

0 and 5       # 0 (first falsy)
3 and 5       # 5 (all truthy, returns last)
1 and 2 and 3 # 3 (returns last truthy value)
```

#### `or` Short-Circuit
- Returns **first truthy value** OR **last value** if all falsy
- Stops at first `True`

```python
True or expensive_function()  # Returns True, function NOT called
False or False or True or print("Never runs")  # Stops at True

0 or 5        # 5 (first truthy)
0 or ""       # "" (all falsy, returns last)
1 or 2 or 3   # 1 (returns first truthy value)
```

#### Practical Use Cases
```python
# Default values
name = user_input or "Guest"  # If user_input is empty, use "Guest"

# Guard clauses
if user and user.is_active and user.has_permission():
    # Stops checking if user is None

# Conditional execution
is_admin and send_admin_email()  # Only calls if is_admin is True
```

### Logical Operators Truth Tables
| A | B | A and B | A or B | not A |
|---|---|---------|--------|-------|
| T | T | **T** | T | F |
| T | F | F | **T** | F |
| F | T | F | **T** | **T** |
| F | F | F | F | **T** |

**Quick Rules:**
- `and` → True only if **both** are True
- `or` → True if **at least one** is True  
- `not` → Reverses the boolean value

### `is` vs `==` (⚠️ Common Trap)
| Operator | Checks | Use When |
|----------|--------|----------|
| `==` | **Value equality** | Comparing values |
| `is` | **Identity (same object)** | Comparing with `None` |

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b   # True (same values)
a is b   # False (different objects in memory)
a is c   # True (same object)

# Always use 'is' for None
x = None
if x is None:    # ✅ Correct
if x == None:    # ⚠️ Works but not Pythonic
```

**Integer Caching (Tricky!):**
```python
a = 256; b = 256
a is b  # True (Python caches -5 to 256)

a = 257; b = 257
a is b  # False (outside cache range)
```

---

## 4. Strings
**Immutable, indexable:** `s[0]`, `s[-1]`, `s[1:4]`  
**Slicing:** `s[start:end:step]` → `s[::-1]` reverses string

### Key String Methods  
| Method | Syntax | Description |
|--------|--------|-------------|
| `upper()` | `s.upper()` | Convert to UPPERCASE |
| `lower()` | `s.lower()` | Convert to lowercase |
| `strip()` | `s.strip()` | Remove leading/trailing whitespace |
| `lstrip()` | `s.lstrip()` | Remove leading whitespace only |
| `rstrip()` | `s.rstrip()` | Remove trailing whitespace only |
| `split()` | `s.split(sep)` | Split by separator (default: space) |
| `join()` | `sep.join(list)` | Join list items with separator |
| `replace()` | `s.replace(old, new)` | Replace all occurrences |
| `find()` | `s.find(sub)` | Find index (returns -1 if not found) |
| `index()` | `s.index(sub)` | Find index (raises error if not found) |
| `count()` | `s.count(sub)` | Count occurrences |
| `startswith()` | `s.startswith(prefix)` | Check if starts with prefix |
| `endswith()` | `s.endswith(suffix)` | Check if ends with suffix |
| `capitalize()` | `s.capitalize()` | Capitalize first character |
| `title()` | `s.title()` | Title Case Each Word |
| `isalpha()` | `s.isalpha()` | Check if all alphabetic |
| `isdigit()` | `s.isdigit()` | Check if all digits |
| `isalnum()` | `s.isalnum()` | Check if alphanumeric |
| `islower()` | `s.islower()` | Check if all lowercase |
| `isupper()` | `s.isupper()` | Check if all uppercase |
| `isspace()` | `s.isspace()` | Check if all whitespace |

```python
s = "hello"; s[::-1]  # 'olleh'
"a,b,c".split(",")    # ['a', 'b', 'c']
"-".join(['a','b'])   # 'a-b'
```

### String Formatting (3 Methods)
```python
name, age = "Vedant", 21

# 1. f-strings (Python 3.6+) ✅ PREFERRED
f"Name: {name}, Age: {age}"           # "Name: Vedant, Age: 21"
f"{age * 2}"                           # "42" (expressions allowed)
f"{3.14159:.2f}"                       # "3.14" (formatting)

# 2. .format() method
"Name: {}, Age: {}".format(name, age)  # positional
"Name: {n}, Age: {a}".format(n=name, a=age)  # named

# 3. % operator (old style)
"Name: %s, Age: %d" % (name, age)
```

**Common Format Specifiers:**
| Specifier | Meaning | Example |
|-----------|---------|----------|
| `:.2f` | 2 decimal places | `f"{3.14159:.2f}"` → `"3.14"` |
| `:05d` | Pad with zeros | `f"{42:05d}"` → `"00042"` |
| `:>10` | Right align (10 chars) | `f"{'hi':>10}"` → `"        hi"` |
| `:<10` | Left align | `f"{'hi':<10}"` → `"hi        "` |
| `:^10` | Center align | `f"{'hi':^10}"` → `"    hi    "` |

---

## 5. Lists
**Ordered, mutable sequence:** can change after creation

### Key List Methods  
| Method | Syntax | Description |
|--------|--------|-------------|
| `append()` | `l.append(item)` | Add item at end |
| `insert()` | `l.insert(index, item)` | Insert at specific index |
| `extend()` | `l.extend(iterable)` | Add multiple items |
| `remove()` | `l.remove(item)` | Remove first occurrence |
| `pop()` | `l.pop(index)` | Remove & return at index (default: -1) |
| `clear()` | `l.clear()` | Remove all items |
| `index()` | `l.index(item)` | Find index of first occurrence |
| `count()` | `l.count(item)` | Count occurrences |
| `sort()` | `l.sort(reverse=False)` | Sort in place |
| `reverse()` | `l.reverse()` | Reverse in place |
| `copy()` | `l.copy()` | Shallow copy of list |

```python
l = [1, 2]; l.append(3)       # [1, 2, 3]
l.insert(1, 5)                # [1, 5, 2, 3]
l.extend([6, 7])              # [1, 5, 2, 3, 6, 7]
l.pop()                       # removes 7, returns 7
sorted(l)                     # returns sorted, doesn't modify l
```

---

## 6. Tuples
**Ordered, immutable:** cannot change after creation  
**Use cases:** fixed records, dictionary keys, unpacking

```python
t = (1, 2, 3)
a, b, c = t       # unpacking
t[0]              # access: 1
t.count(2)        # count occurrences
t.index(3)        # find index
```

**Methods:** `count()`, `index()` only (immutable = fewer methods)

---

## 7. Sets
**Unordered, unique elements:** automatically removes duplicates  
**Fast membership checks:** `item in set` is O(1)

### Key Set Methods  
| Method | Syntax | Description |
|--------|--------|-------------|
| `add()` | `s.add(item)` | Add single element |
| `update()` | `s.update(iterable)` | Add multiple elements |
| `remove()` | `s.remove(item)` | Remove (error if not found) |
| `discard()` | `s.discard(item)` | Remove (no error if not found) |
| `pop()` | `s.pop()` | Remove random element |
| `clear()` | `s.clear()` | Remove all elements |
| `union()` | `s1.union(s2)` or `s1 \| s2` | All elements from both |
| `intersection()` | `s1.intersection(s2)` or `s1 & s2` | Common elements |
| `difference()` | `s1.difference(s2)` or `s1 - s2` | In s1 but not s2 |
| `symmetric_difference()` | `s1.symmetric_difference(s2)` or `s1 ^ s2` | In either but not both |
| **`intersection_update()`** | `s1.intersection_update(s2)` | Update s1 with intersection (modifies s1) |
| **`difference_update()`** | `s1.difference_update(s2)` | Update s1 with difference (modifies s1) |
| **`symmetric_difference_update()`** | `s1.symmetric_difference_update(s2)` | Update s1 with symmetric diff (modifies s1) |

```python
s = {1, 2, 2, 3}  # {1, 2, 3} - auto removes duplicates
s.add(4)          # {1, 2, 3, 4}
{1, 2} & {2, 3}   # {2} - intersection

# Mutation variants (modify in place)
s1 = {1, 2, 3}
s1.intersection_update({2, 3, 4})  # s1 becomes {2, 3}
```

---

## 8. Dictionaries
**Key→value mapping:** keys must be immutable (str, int, tuple)

### Key Dictionary Methods  
| Method | Syntax | Description |
|--------|--------|-------------|
| `get()` | `d.get(key, default)` | Get value (no error if missing) |
| `keys()` | `d.keys()` | View of all keys |
| `values()` | `d.values()` | View of all values |
| `items()` | `d.items()` | View of (key, value) pairs |
| `update()` | `d.update(other_dict)` | Merge dictionaries |
| `pop()` | `d.pop(key, default)` | Remove & return value |
| `popitem()` | `d.popitem()` | Remove last inserted (k, v) |
| `clear()` | `d.clear()` | Remove all items |
| `setdefault()` | `d.setdefault(key, default)` | Get or set if missing |
| `fromkeys()` | `dict.fromkeys(keys, value)` | Create dict from keys (class method) |

```python
d = {'a': 1, 'b': 2}
d.get('c', 0)         # 0 (default)
d.update({'c': 3})    # {'a': 1, 'b': 2, 'c': 3}
for k, v in d.items(): print(k, v)

# fromkeys() - create dict with same value for all keys
keys = ['name', 'age', 'city']
default_dict = dict.fromkeys(keys, 'N/A')  # {'name': 'N/A', 'age': 'N/A', 'city': 'N/A'}
zero_dict = dict.fromkeys(['a', 'b', 'c'], 0)  # {'a': 0, 'b': 0, 'c': 0}
```

---

## 9. Conditional Statements
```python
if condition:
    # code
elif other_condition:
    # code
else:
    # code

# Ternary operator
result = value_if_true if condition else value_if_false
```

```python
age = 20
status = 'Adult' if age >= 18 else 'Minor'
```

---

## 10. Loops

### `for` Loop
**Syntax:** `for item in iterable:`  
**Use:** Iterate over sequences (lists, strings, ranges, dicts)

```python
for i in range(5):        # 0, 1, 2, 3, 4
for i in range(2, 7):     # 2, 3, 4, 5, 6
for i in range(0, 10, 2): # 0, 2, 4, 6, 8 (step=2)
```

**`range()` parameters:** `range(start=0, stop, step=1)`
- `range(5)` → 0 to 4
- `range(2, 5)` → 2, 3, 4
- `range(0, 10, 2)` → 0, 2, 4, 6, 8

### `while` Loop
```python
while condition:
    # code
```

### Loop Controls
- `break` → exit loop immediately
- `continue` → skip to next iteration
- `else` → executes if loop completes without `break`

```python
for i in range(5):
    if i == 3: break
    print(i)  # 0, 1, 2
```

---

## 11. Comprehensions
**Fast, concise creation** of lists/dicts/sets

```python
# List comprehension
[expression for item in iterable if condition]
squares = [x**2 for x in range(5) if x % 2 == 0]  # [0, 4, 16]

# Dict comprehension
{key_expr: value_expr for item in iterable}
{x: x**2 for x in range(3)}  # {0: 0, 1: 1, 2: 4}

# Set comprehension
{expression for item in iterable}
{x % 3 for x in range(10)}  # {0, 1, 2}
```

---

## 12. Functions

### Basic Syntax
```python
def function_name(parameters):
    """Docstring: explains function"""
    # code
    return value
```

### Parameter Types
```python
def func(a, b=10, *args, **kwargs):
    # a: positional
    # b: default value
    # *args: variable positional (tuple)
    # **kwargs: variable keyword (dict)
    pass

func(1)              # a=1, b=10
func(1, 20)          # a=1, b=20
func(1, 2, 3, 4)     # a=1, b=2, args=(3, 4)
func(1, x=5, y=10)   # a=1, b=10, kwargs={'x': 5, 'y': 10}
```

### Lambda Functions
**Anonymous functions:** `lambda args: expression`

```python
square = lambda x: x ** 2
square(5)  # 25

# With map/filter
list(map(lambda x: x*2, [1, 2, 3]))     # [2, 4, 6]
list(filter(lambda x: x > 2, [1, 2, 3, 4]))  # [3, 4]
```

### Variable Scope
- **Local:** Inside function
- **Global:** Module level (use `global` keyword to modify)
- **Enclosing:** In nested functions (use `nonlocal` for nested)

```python
x = 10  # global
def modify():
    global x
    x += 1  # modifies global x
```

---

## 13. Iterators & Generators

### Iterator vs Iterable
| Term | Definition | Example |
|------|------------|----------|
| **Iterable** | Has `__iter__()` method | `list`, `str`, `dict` |
| **Iterator** | Has `__iter__()` AND `__next__()` | `iter([1,2,3])` |

```python
# Creating iterator from iterable
my_list = [1, 2, 3]
my_iter = iter(my_list)  # Create iterator

next(my_iter)  # 1
next(my_iter)  # 2
next(my_iter)  # 3
next(my_iter)  # StopIteration error
```

### Custom Iterator
```python
class Counter:
    def __init__(self, max):
        self.max = max
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max:
            self.current += 1
            return self.current
        raise StopIteration

for num in Counter(3):
    print(num)  # 1, 2, 3
```

### Generators (Simpler Way)
**Use `yield` instead of `return`** — pauses and resumes

```python
def countdown(n):
    while n > 0:
        yield n  # pauses here, returns value
        n -= 1

for num in countdown(3):
    print(num)  # 3, 2, 1

# Generator expression (like list comprehension)
gen = (x**2 for x in range(5))  # generator object
list(gen)  # [0, 1, 4, 9, 16]
```

**Why Generators?**
- **Memory efficient:** Don't store all values at once
- **Lazy evaluation:** Compute values on-demand
- **Infinite sequences:** Can represent infinite data

```python
# Infinite generator
def infinite_counter():
    n = 0
    while True:
        yield n
        n += 1
```

---

## 14. Decorators
**Functions that modify other functions' behavior**

### Basic Decorator
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function")
        result = func(*args, **kwargs)
        print("After function")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("Vedant")
# Output:
# Before function
# Hello, Vedant!
# After function
```

**`@decorator` is equivalent to:** `func = decorator(func)`

### Preserving Function Metadata with `functools.wraps` ⚠️
```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # ✅ Preserves __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greet someone."""
    return f"Hello, {name}!"

print(greet.__name__)  # 'greet' (without @wraps: 'wrapper')
print(greet.__doc__)   # 'Greet someone.' (without @wraps: None)
```

### Decorator with Arguments
```python
from functools import wraps

def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("Hello!")

say_hello()  # Prints "Hello!" 3 times
```

### Common Built-in Decorators
| Decorator | Purpose |
|-----------|----------|
| `@staticmethod` | No `self` or `cls` access |
| `@classmethod` | Access `cls` (class) |
| `@property` | Getter method as attribute |
| `@abstractmethod` | Must be implemented by subclass |

### `@property` Decorator
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):       # getter
        return self._radius
    
    @radius.setter
    def radius(self, value): # setter
        if value > 0:
            self._radius = value
    
    @property
    def area(self):         # computed property
        return 3.14 * self._radius ** 2

c = Circle(5)
c.radius      # 5 (uses getter)
c.radius = 10 # uses setter
c.area        # 314.0 (computed)
```

---

## 15. OOP Basics
**Class:** Blueprint for objects  
**Instance:** Actual object created from class  
**`__init__`:** Constructor (runs when creating object)  
**`self`:** Refers to instance itself

```python
class Person:
    def __init__(self, name, age):
        self.name = name  # instance variable
        self.age = age
    
    def greet(self):
        return f"Hi, I'm {self.name}"

p = Person("Vedant", 21)
p.greet()  # "Hi, I'm Vedant"
```

---
## 16. Inheritance & Method Overriding
**Child class overrides parent method**  
Use `super()` to call parent implementation

```python
class Parent:
    def greet(self):
        return "Hello from Parent"

class Child(Parent):
    def greet(self):
        parent_msg = super().greet()  # call parent
        return f"{parent_msg} and Child"
```

---

### Inheritance Types

#### Single Inheritance
```python
class Parent:
    pass
class Child(Parent):  # One parent
    pass
```

#### Multiple Inheritance
```python
class A:
    pass
class B:
    pass
class C(A, B):  # Multiple parents
    pass
```

#### Multilevel Inheritance
```python
class Grandparent:
    pass
class Parent(Grandparent):
    pass
class Child(Parent):  # Chain: Grandparent → Parent → Child
    pass
```

#### Hierarchical Inheritance
```python
class Parent:
    pass
class Child1(Parent):  # Multiple children, one parent
    pass
class Child2(Parent):
    pass
```

#### Hybrid Inheritance
Combination of two or more inheritance types.

#### MRO (Method Resolution Order) ⚠️ Important
```python
class A:
    def show(self): print("A")
class B(A):
    def show(self): print("B")
class C(A):
    def show(self): print("C")
class D(B, C):  # Diamond problem
    pass

print(D.mro())  # [D, B, C, A, object]
D().show()      # "B" (follows MRO: left to right)
```

**MRO Rule:** Left-to-right, depth-first, but each class appears only once.

---

## 17. Encapsulation
**Control access** using naming conventions:
- `public` → normal: `self.name`
- `_protected` → convention: `self._name` (still accessible)
- `__private` → name mangling: `self.__name` (harder to access)

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # private
    
    def get_balance(self):  # getter
        return self.__balance
    
    def deposit(self, amount):  # setter with validation
        if amount > 0:
            self.__balance += amount
```

---

## 18. Abstraction
**Hide implementation details**, show only necessary interface  
Use `abc` module for abstract base classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass  # must be implemented by subclass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):  # implementation
        return 3.14 * self.radius ** 2
```

---

## 19. Polymorphism
**Same interface, different implementations**  
Duck typing: "If it quacks like a duck..."

```python
class Dog:
    def speak(self): return "Woof"

class Cat:
    def speak(self): return "Meow"

for animal in [Dog(), Cat()]:
    print(animal.speak())  # Different behavior, same method
```

---

## 20. Instance/Class/Static Methods

```python
class MyClass:
    class_var = 10  # class variable
    
    def instance_method(self):
        # has access to self (instance)
        return self.class_var
    
    @classmethod
    def class_method(cls):
        # has access to cls (class)
        return cls.class_var
    
    @staticmethod
    def static_method():
        # no access to self or cls
        return "static"
```

---

## 21. Dunder/Magic Methods
**Double underscore methods** that Python calls automatically.

### Most Important Dunder Methods
| Method | Called When | Example |
|--------|-------------|----------|
| `__init__(self)` | Object created | `obj = MyClass()` |
| `__str__(self)` | `str(obj)` or `print(obj)` | Human-readable |
| `__repr__(self)` | `repr(obj)` or in REPL | Developer-readable |
| `__len__(self)` | `len(obj)` | Return length |
| `__eq__(self, other)` | `obj1 == obj2` | Equality check |
| `__lt__(self, other)` | `obj1 < obj2` | Less than |
| `__add__(self, other)` | `obj1 + obj2` | Addition |
| `__getitem__(self, key)` | `obj[key]` | Indexing |
| `__setitem__(self, key, val)` | `obj[key] = val` | Assignment |
| `__iter__(self)` | `for x in obj` | Make iterable |
| `__call__(self)` | `obj()` | Make callable |

```python
class Book:
    def __init__(self, title, pages):
        self.title = title
        self.pages = pages
    
    def __str__(self):      # for print()
        return f"{self.title}"
    
    def __repr__(self):     # for debugging
        return f"Book('{self.title}', {self.pages})"
    
    def __len__(self):      # for len()
        return self.pages
    
    def __eq__(self, other): # for ==
        return self.title == other.title

b = Book("Python", 300)
print(b)      # Python (__str__)
len(b)        # 300 (__len__)
```

---

## 22. File Handling

### File Modes
| Mode | Description | Creates if missing? |
|------|-------------|---------------------|
| `'r'` | Read (default) | No, error if not found |
| `'w'` | Write (overwrites) | Yes |
| `'a'` | Append | Yes |
| `'r+'` | Read + Write | No |
| `'b'` | Binary mode | Add to others (e.g., `'rb'`) |

### File Methods  
| Method | Description |
|--------|-------------|
| `read()` | Read entire file |
| `read(n)` | Read n characters |
| `readline()` | Read one line |
| `readlines()` | Read all lines as list |
| `write(text)` | Write text |
| `writelines(list)` | Write list of strings |
| `close()` | Close file (auto with `with`) |

```python
# Best practice: with statement (auto-closes)
with open('file.txt', 'r') as f:
    data = f.read()

# Writing
with open('output.txt', 'w') as f:
    f.write("Hello World\n")
```

---

## 23. Exception Handling

```python
try:
    # code that might raise exception
    result = 10 / 0
except ZeroDivisionError as e:
    # handle specific exception
    print(f"Error: {e}")
except Exception as e:
    # handle any exception
    print(f"General error: {e}")
else:
    # runs if no exception
    print("Success!")
finally:
    # always runs (cleanup)
    print("Cleanup")

# Raise exceptions
raise ValueError("Invalid value")
```

**Common exceptions:** `ValueError`, `TypeError`, `KeyError`, `IndexError`, `ZeroDivisionError`, `FileNotFoundError`

### Exception Groups (Python 3.11+)
```python
# Raise multiple exceptions at once
def process_items(items):
    errors = []
    for item in items:
        try:
            validate(item)
        except ValueError as e:
            errors.append(e)
    if errors:
        raise ExceptionGroup("Validation failed", errors)

# Handle with except*
try:
    process_items([1, -1, "bad"])
except* ValueError as eg:
    print(f"Value errors: {eg.exceptions}")
except* TypeError as eg:
    print(f"Type errors: {eg.exceptions}")
```

### Custom Exceptions
```python
class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

raise ValidationError("email", "Invalid format")
```

---

## 24. Shallow vs Deep Copy

### The Problem
```python
original = [[1, 2], [3, 4]]
shallow = original.copy()  # or list(original)

shallow[0][0] = 99
print(original)  # [[99, 2], [3, 4]] ← Original also changed!
```

### Solution: Deep Copy
```python
import copy

original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)

deep[0][0] = 99
print(original)  # [[1, 2], [3, 4]] ← Original unchanged!
```

### Comparison Table
| Type | Syntax | Nested Objects | Use When |
|------|--------|----------------|----------|
| **Assignment** | `b = a` | Same object | Aliasing |
| **Shallow Copy** | `a.copy()`, `list(a)`, `a[:]` | Shared (reference) | Flat structures |
| **Deep Copy** | `copy.deepcopy(a)` | Independent copies | Nested structures |

```python
import copy

a = [1, [2, 3]]

# Assignment (alias)
b = a
b is a  # True (same object)

# Shallow copy
c = a.copy()
c is a      # False (different list)
c[1] is a[1]  # True (nested list shared!)

# Deep copy
d = copy.deepcopy(a)
d is a      # False
d[1] is a[1]  # False (completely independent)
```

---

## 25. Type Hints
**Static type annotations** for better code documentation and IDE support.

### Basic Type Hints
```python
# Variables
name: str = "Vedant"
age: int = 21
price: float = 9.99
is_active: bool = True

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

# None return type
def log(message: str) -> None:
    print(message)
```

### Collection Type Hints
```python
# Use built-in types directly (Python 3.10+)
numbers: list[int] = [1, 2, 3]
scores: dict[str, int] = {"alice": 100}
coords: tuple[int, int] = (10, 20)
unique: set[str] = {"a", "b"}

# Nested collections
matrix: list[list[int]] = [[1, 2], [3, 4]]
user_data: dict[str, list[int]] = {"scores": [90, 85, 88]}
```

### Optional & Union (Python 3.10+ syntax)
```python
# Optional = can be None (use | None)
def find_user(id: int) -> str | None:
    return None if id < 0 else "User"

# Union = multiple possible types (use |)
def process(value: int | str) -> str:
    return str(value)

# Multiple types
def handle(data: int | float | str | None) -> str:
    return str(data) if data else "Empty"
```

### Callable & TypeVar
```python
from typing import Callable, TypeVar

# Function type hints
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Generic type variable
T = TypeVar('T')
def first(items: list[T]) -> T:
    return items[0]
```

### Type Aliases
```python
type Vector = list[float]  # Python 3.12+ syntax
type Matrix = list[Vector]

# Python 3.10-3.11
from typing import TypeAlias
Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]

def scale(v: Vector, factor: float) -> Vector:
    return [x * factor for x in v]
```

### Self Type (Python 3.11+)
```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self  # Enables method chaining
    
    def set_age(self, age: int) -> Self:
        self.age = age
        return self

# Usage
builder = Builder().set_name("Vedant").set_age(21)
```

---

## 26. Dataclasses
**Automatic generation** of `__init__`, `__repr__`, `__eq__`, etc.

### Basic Dataclass
```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str = ""  # default value

# Auto-generated __init__, __repr__, __eq__
p = Person("Vedant", 21)
print(p)  # Person(name='Vedant', age=21, email='')

p1 = Person("Vedant", 21)
p2 = Person("Vedant", 21)
p1 == p2  # True (auto __eq__)
```

### Dataclass Options
```python
from dataclasses import dataclass, field

@dataclass(frozen=True)  # Immutable (hashable)
class Point:
    x: int
    y: int

@dataclass(order=True)  # Adds comparison methods
class Student:
    name: str
    grade: float

@dataclass(slots=True)  # Memory efficient (Python 3.10+)
class User:
    name: str
    age: int
    # Uses __slots__ instead of __dict__ → less memory, faster access

# Mutable default values - use field()
@dataclass
class Team:
    name: str
    members: list = field(default_factory=list)  # ✅ Correct
    # members: list = []  # ❌ WRONG! Same mutable default trap
```

### Dataclass Options Reference
| Option | Default | Purpose |
|--------|---------|----------|
| `frozen` | `False` | Make immutable (hashable) |
| `order` | `False` | Add `<`, `>`, `<=`, `>=` methods |
| `slots` | `False` | Use `__slots__` for memory efficiency |
| `kw_only` | `False` | All fields keyword-only (3.10+) |
| `match_args` | `True` | Enable pattern matching support |

### Post-Init Processing
```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Not in __init__
    
    def __post_init__(self):
        self.area = self.width * self.height

r = Rectangle(10, 5)
r.area  # 50.0 (computed automatically)
```

### Dataclass vs NamedTuple vs Regular Class
| Feature | `@dataclass` | `NamedTuple` | Regular Class |
|---------|--------------|--------------|---------------|
| Mutable | ✅ (default) | ❌ | ✅ |
| Auto `__init__` | ✅ | ✅ | ❌ |
| Auto `__repr__` | ✅ | ✅ | ❌ |
| Type hints | ✅ Required | ✅ Required | Optional |
| Inheritance | ✅ | Limited | ✅ |
| Memory | Normal | Optimized | Normal |

---

## 27. Match Statement
**Structural pattern matching** — like switch/case on steroids.

### Basic Match
```python
def http_status(status: int) -> str:
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:  # default (wildcard)
            return "Unknown"

http_status(200)  # "OK"
http_status(999)  # "Unknown"
```

### Pattern Matching with OR
```python
match status:
    case 200 | 201 | 204:  # Multiple values
        return "Success"
    case 400 | 401 | 403 | 404:
        return "Client Error"
    case _:
        return "Other"
```

### Matching Sequences
```python
def analyze(point):
    match point:
        case (0, 0):
            return "Origin"
        case (0, y):  # Capture y
            return f"On Y-axis at {y}"
        case (x, 0):  # Capture x
            return f"On X-axis at {x}"
        case (x, y):  # Capture both
            return f"Point at ({x}, {y})"
        case _:
            return "Not a point"

analyze((0, 5))  # "On Y-axis at 5"
```

### Matching Dictionaries
```python
def process_event(event: dict):
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"
        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"
        case {"type": type_}:  # Capture unknown type
            return f"Unknown event: {type_}"
```

### Matching with Guards
```python
match point:
    case (x, y) if x == y:
        return "On diagonal"
    case (x, y) if x > 0 and y > 0:
        return "First quadrant"
    case (x, y):
        return f"Point at ({x}, {y})"
```

### Matching Classes
```python
@dataclass
class Point:
    x: int
    y: int

def locate(point):
    match point:
        case Point(x=0, y=0):
            return "Origin"
        case Point(x=0, y=y):
            return f"Y-axis at {y}"
        case Point(x=x, y=0):
            return f"X-axis at {x}"
        case Point():
            return "Somewhere else"
```

---

## 28. Walrus Operator `:=`
**Assignment expression** — assign and use in one expression.

### Basic Usage
```python
# Without walrus operator
data = get_data()
if data:
    process(data)

# With walrus operator ✅
if (data := get_data()):
    process(data)
```

### Common Use Cases
```python
# 1. While loops with assignment
while (line := file.readline()):
    process(line)

# 2. List comprehensions with expensive computation
# Without walrus - calls compute() twice:
[compute(x) for x in data if compute(x) > 0]

# With walrus - calls compute() once: ✅
[y for x in data if (y := compute(x)) > 0]

# 3. Regex matching
import re
if (match := re.search(r'\d+', text)):
    print(f"Found: {match.group()}")

# 4. Avoiding repeated dictionary lookups
if (value := my_dict.get('key')) is not None:
    process(value)
```

### When NOT to Use ⚠️
```python
# Don't use for simple assignments
x := 5  # ❌ SyntaxError! Must be in parentheses or expression

# Don't overuse - readability matters
result = (a := 1) + (b := 2) + (c := 3)  # ⚠️ Hard to read

# Keep it simple
a, b, c = 1, 2, 3  # ✅ Clearer
result = a + b + c
```

---

## 29. Context Managers (`with` Statement)
**Automatic resource management** — ensures cleanup even if errors occur.

### Built-in Context Managers
```python
# File handling (auto-closes)
with open('file.txt', 'r') as f:
    data = f.read()
# f is automatically closed here

# Multiple context managers
with open('in.txt') as f_in, open('out.txt', 'w') as f_out:
    f_out.write(f_in.read())

# Threading locks
import threading
lock = threading.Lock()
with lock:
    # Critical section
    pass
```

### Creating Context Managers

#### Method 1: Class-based (`__enter__` / `__exit__`)
```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self  # Returned to 'as' variable
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end = time.time()
        print(f"Elapsed: {self.end - self.start:.2f}s")
        return False  # Don't suppress exceptions

with Timer() as t:
    # code to time
    pass
```

#### Method 2: Generator-based (`@contextmanager`) ✅ Simpler
```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield  # Code inside 'with' runs here
    finally:
        end = time.time()
        print(f"Elapsed: {end - start:.2f}s")

with timer():
    # code to time
    pass
```

### `__exit__` Parameters
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    # exc_type: Exception class (e.g., ValueError)
    # exc_val: Exception instance
    # exc_tb: Traceback object
    
    if exc_type is ValueError:
        print("Caught ValueError, suppressing...")
        return True  # Suppress the exception
    return False  # Re-raise any other exception
```

---

## 30. Async/Await
**Asynchronous programming** for I/O-bound concurrent tasks.

### Basic Concepts
| Term | Meaning |
|------|---------|  
| `async def` | Defines a coroutine function |
| `await` | Pause until coroutine completes |
| `asyncio` | Standard library for async I/O |
| Coroutine | Object returned by async function |
| Event Loop | Runs and manages coroutines |

### Basic Async Function
```python
import asyncio

async def fetch_data():
    print("Fetching...")
    await asyncio.sleep(2)  # Non-blocking sleep
    print("Done!")
    return {"data": 123}

# Running the coroutine
async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())  # Entry point
```

### Running Tasks Concurrently
```python
import asyncio

async def task(name, delay):
    print(f"{name} starting")
    await asyncio.sleep(delay)
    print(f"{name} done")
    return name

async def main():
    # Run concurrently (not sequentially!)
    results = await asyncio.gather(
        task("A", 2),
        task("B", 1),
        task("C", 3)
    )
    print(results)  # ['A', 'B', 'C']

asyncio.run(main())  # Total time: ~3s (not 6s)
```

### Creating Tasks
```python
async def main():
    # Create task (starts immediately)
    task1 = asyncio.create_task(fetch_data())
    task2 = asyncio.create_task(fetch_data())
    
    # Do other work while tasks run...
    
    # Wait for results
    result1 = await task1
    result2 = await task2
```

### Async Context Managers & Iterators
```python
# Async context manager
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.text()

# Async iterator
async for item in async_generator():
    process(item)
```

### When to Use Async ⚠️
| Use Async For | Don't Use Async For |
|---------------|--------------------|
| Network I/O (HTTP, databases) | CPU-bound tasks |
| File I/O (with async libs) | Simple scripts |
| Many concurrent connections | Single operations |
| Web servers/APIs | Heavy computations |

**For CPU-bound:** Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`

---

## 31. Common Traps

### Mutable Default Arguments ⚠️
```python
def bad_func(items=[]):  # ❌ WRONG!
    items.append(1)
    return items

bad_func()  # [1]
bad_func()  # [1, 1] ← Same list reused!

def good_func(items=None):  # ✅ CORRECT
    if items is None:
        items = []
    items.append(1)
    return items
```

### List Aliasing
```python
a = [1, 2, 3]
b = a        # b is alias (same object)
b.append(4)
print(a)     # [1, 2, 3, 4] ← a also changed!

# Fix: Create copy
b = a.copy()  # or a[:] or list(a)
```

### Integer Caching
```python
a = 256; b = 256
a is b  # True (cached: -5 to 256)

a = 257; b = 257  
a is b  # False (outside cache)
a == b  # True (values equal)
```

### String Immutability
```python
s = "hello"
s[0] = "H"  # ❌ TypeError! Strings are immutable
s = "H" + s[1:]  # ✅ Create new string
```

### `for` Loop Variable Scope
```python
for i in range(3):
    pass
print(i)  # 2 ← Variable exists after loop!
```

### Truthy/Falsy Values
```python
# Falsy values (evaluate to False)
bool(0)       # False
bool(0.0)     # False
bool("")      # False (empty string)
bool([])      # False (empty list)
bool({})      # False (empty dict)
bool(None)    # False

# Everything else is Truthy
bool(1)       # True
bool("hello") # True
bool([0])     # True (non-empty list)
```

### `pass` vs `continue` vs `break`
| Keyword | Purpose |
|---------|---------|
| `pass` | Do nothing (placeholder) |
| `continue` | Skip to next iteration |
| `break` | Exit loop entirely |

### Dictionary Key Overwriting
```python
d = {'a': 1, 'a': 2}  # Duplicate key!
print(d)  # {'a': 2} ← Last value wins
```

---

## 32. Quick Reference

### Useful Snippets
```python
# Swap
a, b = b, a

# Check empty
if not my_list:  # True if empty

# Reverse
s[::-1]

# Merge dicts (Python 3.9+)
{**dict1, **dict2}    # or: dict1 | dict2

# Unique elements (preserve order)
list(dict.fromkeys(seq))

# Flatten nested list
[item for sublist in nested for item in sublist]

# Enumerate with index
for i, item in enumerate(my_list):
    print(i, item)

# Zip two lists
list(zip([1, 2], ['a', 'b']))  # [(1, 'a'), (2, 'b')]

# Any/All
any([False, True, False])  # True (at least one True)
all([True, True, False])   # False (not all True)

# Get with default
value = my_dict.get('key', 'default')

# Conditional expression
result = 'yes' if condition else 'no'

# Safe division
result = a // b if b else 0

# Group by key (must sort by same key first!)
from itertools import groupby
key_func = lambda x: x[0]
{k: list(v) for k, v in groupby(sorted(items, key=key_func), key=key_func)}
```

### Useful Standard Library Modules (Python 3.11+)
```python
# tomllib - Parse TOML files (Python 3.11+)
import tomllib
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# pathlib - Modern file paths
from pathlib import Path
path = Path("dir") / "file.txt"  # Cross-platform
path.read_text()                  # Read file
path.exists()                     # Check existence
list(Path(".").glob("*.py"))      # Find files

# functools - Function utilities
from functools import cache, partial
@cache  # Memoization (simpler than lru_cache)
def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)

# itertools - Iterator utilities
from itertools import chain, groupby, combinations
list(chain([1,2], [3,4]))         # [1, 2, 3, 4]
list(combinations([1,2,3], 2))    # [(1,2), (1,3), (2,3)]

# collections - Specialized containers
from collections import Counter, defaultdict, deque
Counter("hello")                  # {'l': 2, 'h': 1, 'e': 1, 'o': 1}
defaultdict(list)                 # Dict with default factory
deque([1,2,3], maxlen=3)          # Fixed-size queue
```

---
