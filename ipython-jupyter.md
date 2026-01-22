# 🐍 IPython & Jupyter Cheatsheet

**Author:** @VedantAndhale  
**Purpose:** Comprehensive yet scannable IPython/Jupyter notes with commands, magic functions, and productivity tips.

---

## Table of Contents
1. [IPython Basics](#1-ipython-basics)
2. [Getting Help](#2-getting-help)
3. [Magic Commands](#3-magic-commands)
4. [Shell Commands](#4-shell-commands)
5. [Input/Output History](#5-inputoutput-history)
6. [Keyboard Shortcuts](#6-keyboard-shortcuts)
7. [Display & Rich Output](#7-display--rich-output)
8. [Debugging](#8-debugging)
9. [Configuration & Profiles](#9-configuration--profiles)
10. [Common Traps](#10-common-traps)
11. [Quick Reference](#11-quick-reference)

---

## 1. IPython Basics

**IPython** = **I**nteractive **Python**  
- Enhanced interactive Python shell
- Foundation for **Jupyter Notebooks**
- Created by **Fernando Pérez** in 2001

### Launch IPython
```bash
ipython              # Start IPython shell
jupyter notebook     # Start Jupyter Notebook
jupyter lab          # Start JupyterLab (modern UI)
```

### IPython vs Python REPL
| Feature | Python REPL | IPython |
|---------|-------------|---------|
| Syntax highlighting | ❌ | ✅ |
| Tab completion | Basic | Advanced |
| Magic commands | ❌ | ✅ |
| Shell integration | ❌ | ✅ |
| Rich output (HTML, images) | ❌ | ✅ |
| History search | Basic | `Ctrl+R` fuzzy search |

---

## 2. Getting Help

**If you forget how a function works, ask Python directly.**

### Help Commands

| Command | Action | Example |
|---------|--------|---------|
| `?` | Quick Help: Shows docstring/description | `len?` or `obj.func?` |
| `??` | Source Code: Shows actual source code | `len??` |
| `Tab` | Auto-Complete: Shows available methods/attributes | `np.<TAB>` |
| `*?` | Wildcard Search: Finds names matching a pattern | `*Warning?` |

### Quick Help (`?`)

```python
# Append ? to any object/function
len?           # Shows docstring for len()
np.array?      # Shows numpy.array documentation
my_list.append? # Shows list.append() help

# Prefix also works
?len           # Same as len?
```

**Output includes:**
- Signature (parameters)
- Docstring (description)
- Type information

### Source Code (`??`)

```python
# Double ?? shows source code (if available)
np.sum??       # Shows actual implementation

# Note: Built-in C functions won't show source
len??          # Shows docstring only (C implementation)
```

### Tab Completion

```python
# Object attributes and methods
my_list.<TAB>    # Shows: append, clear, copy, count, extend...

# Module contents
np.<TAB>         # Shows all numpy functions

# File paths
open('data/<TAB>  # Auto-completes file names

# Function arguments (Jupyter)
np.array(<TAB>    # Shows parameter hints
```

### Wildcard Search (`*?`)

```python
# Find all names matching pattern
*Warning?        # Finds: DeprecationWarning, UserWarning, etc.
str.*upper*?     # Finds: upper, isupper
np.*sort*?       # Finds: sort, argsort, lexsort, etc.
```

---

## 3. Magic Commands

**Special commands starting with `%` — superpowers for your environment.**

### Line Magic (`%`) vs Cell Magic (`%%`)

| Type | Syntax | Scope |
|------|--------|-------|
| Line magic | `%command` | Operates on single line |
| Cell magic | `%%command` | Operates on entire cell |

### Essential Magic Commands ⭐

| Command | Action |
|---------|--------|
| `%run script.py` | Run external Python script in notebook |
| `%timeit code` | Benchmark single line (~1000 runs) |
| `%%timeit` | Benchmark entire cell (put at top) |
| `%time code` | Time single execution (not averaged) |
| `%%time` | Time entire cell execution |
| `%history` | Show command history |
| `%pwd` | Print working directory |
| `%cd path` | Change directory |
| `%ls` | List files (like shell `ls`) |

### Timing & Benchmarking ⚡

```python
# %timeit - Multiple runs, statistical average
%timeit [x**2 for x in range(1000)]
# Output: 52.3 µs ± 1.2 µs per loop (mean ± std. dev. of 7 runs)

# %%timeit - For multi-line code (MUST be first line)
%%timeit
result = []
for x in range(1000):
    result.append(x**2)

# %time - Single run (good for long operations)
%time big_result = expensive_function()
# Output: CPU times: user 2.5 s, sys: 100 ms, total: 2.6 s

# Compare approaches
%timeit sum(range(1000))        # Python sum
%timeit np.sum(np.arange(1000)) # NumPy sum
```

### Running Scripts

```python
# Run external script
%run my_script.py

# Run with arguments
%run my_script.py arg1 arg2

# Run and enter debugger on error
%run -d my_script.py

# Run in interactive namespace (notebook variables available)
%run -i my_script.py
```

### Working with Files

```python
# Write cell contents to file
%%writefile my_script.py
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()

# Load file contents into cell
%load my_script.py

# Show file contents
%pycat my_script.py
```

### Environment & Variables

```python
# List all variables
%who          # Names only
%whos         # Detailed (name, type, value)
%who_ls       # Returns as list

# Filter by type
%who str      # Only strings
%who int      # Only integers
%who function # Only functions

# Delete variables
%reset        # Delete all (asks confirmation)
%reset -f     # Force delete all
%xdel var     # Delete specific variable
```

### Magic Discovery

```python
%magic        # Full documentation of magic system
%lsmagic      # List all available magics
%quickref     # Quick reference card

# Get help on specific magic
%timeit?      # Shows timeit documentation
```

### "Nice to Have" Magic Commands

| Command | Action |
|---------|--------|
| `%magic` | Print documentation for all magic commands |
| `%lsmagic` | List available magic commands |
| `%automagic` | Toggle needing `%` prefix |
| `%matplotlib inline` | Display plots inline |
| `%load_ext extension` | Load IPython extension |
| `%store var` | Store variable for use in other notebooks |
| `%recall N` | Recall input line N for editing |
| `%save file.py 1-10` | Save lines 1-10 to file |
| `%notebook file.ipynb` | Export history to notebook |

---

## 4. Shell Commands

**Talk to your operating system without leaving Jupyter using `!`.**

### Basic Shell Commands

| Command | Action | Platform |
|---------|--------|----------|
| `!ls` | List files in current folder | Mac/Linux |
| `!dir` | List files in current folder | Windows |
| `!pwd` | Print working directory | All |
| `!cd path` | Change directory (⚠️ subshell only, doesn't persist) | All |
| `!pip install package` | Install Python package | All |
| `!head file.csv` | Show first 10 lines of file | Mac/Linux |
| `!tail file.csv` | Show last 10 lines of file | Mac/Linux |
| `!cat file.txt` | Display file contents | Mac/Linux |
| `!mkdir folder` | Create directory | All |

```python
# List files
!ls -la          # Detailed listing
!ls *.py         # Only Python files

# File operations
!head -n 5 data.csv    # First 5 lines
!wc -l data.csv        # Count lines

# Install packages
!pip install pandas numpy matplotlib
!pip list              # Show installed packages
```

### Capture Shell Output to Python Variable

```python
# Assign shell output to variable
files = !ls
print(files)           # ['file1.py', 'file2.py', ...]
print(type(files))     # <class 'IPython.utils.text.SList'>

# Convert to regular list
file_list = files.list_py()

# Grep-like filtering
py_files = !ls *.py
csv_files = !ls data/*.csv

# Use Python variables in shell commands
filename = "data.csv"
!head -n 5 {filename}  # Use {} for variable interpolation

path = "/home/user"
!ls {path}
```

### Shell Commands vs Magic Commands

```python
# These are equivalent:
!pwd           # Shell command
%pwd           # Magic command (preferred)

!ls            # Shell command
%ls            # Magic command (preferred)

# Magic commands are more portable across platforms
```

---

## 5. Input/Output History

**IPython remembers your previous results.**

### Output History

| Symbol | Meaning |
|--------|---------|
| `_` | Output of the previous cell |
| `__` | Output of the cell before previous |
| `___` | Output of the third-to-last cell |
| `_N` | Output of `Out[N]` (e.g., `_5`) |
| `Out[N]` | Same as `_N` |

```python
# Cell 1
In [1]: 2 + 2
Out[1]: 4

# Cell 2
In [2]: 10 * 5
Out[2]: 50

# Cell 3
In [3]: _       # Previous output
Out[3]: 50

In [4]: _1      # Output of Out[1]
Out[4]: 4

In [5]: __      # Second-to-last output
Out[5]: 50

In [6]: Out[2]  # Same as _2
Out[6]: 50
```

### Input History

| Symbol | Meaning |
|--------|---------|
| `_i` | Previous input (as string) |
| `_ii` | Input before previous |
| `_iN` | Input of `In[N]` (e.g., `_i5`) |
| `In[N]` | Same as `_iN` |

```python
# View input history
%history          # All history
%history -n       # With line numbers
%history 1-5      # Lines 1-5 only
%history -g pattern  # Search history

# Programmatic access
print(In[1])      # First input as string
print(_i)         # Previous input
```

### Suppress Output

```python
# Add semicolon to suppress output display
result = expensive_computation();  # No Out[] shown

# Useful for:
plt.plot(x, y);   # Suppress matplotlib text output
```

---

## 6. Keyboard Shortcuts

### Essential Shortcuts ⭐ (Memorize These!)

| Shortcut | Action | Mode |
|----------|--------|------|
| `Shift + Enter` | Run cell, select next | Both |
| `Ctrl + Enter` | Run cell, stay on cell | Both |
| `Alt + Enter` | Run cell, insert below | Both |
| `Esc` | Enter Command mode | Edit → Command |
| `Enter` | Enter Edit mode | Command → Edit |

### Command Mode (Press `Esc` first)

| Shortcut | Action |
|----------|--------|
| `A` | Insert cell **A**bove |
| `B` | Insert cell **B**elow |
| `D, D` | **D**elete cell (press D twice) |
| `M` | Convert to **M**arkdown |
| `Y` | Convert to code (p**Y**thon) |
| `C` | **C**opy cell |
| `V` | Paste cell below |
| `X` | Cut cell |
| `Z` | Undo cell operation |
| `Shift + M` | **M**erge selected cells |
| `↑` / `↓` | Navigate cells |
| `Shift + ↑/↓` | Select multiple cells |
| `L` | Toggle **L**ine numbers |
| `O` | Toggle **O**utput |
| `H` | Show all shortcuts (**H**elp) |

### Edit Mode (Press `Enter` first)

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete / Indent |
| `Shift + Tab` | Show tooltip (function signature) |
| `Ctrl + ]` | Indent selection |
| `Ctrl + [` | Dedent selection |
| `Ctrl + /` | Comment/uncomment line |
| `Ctrl + D` | Delete line |
| `Ctrl + Z` | Undo |
| `Ctrl + Shift + Z` | Redo |
| `Ctrl + A` | Select all |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl + Home` | Go to cell start |
| `Ctrl + End` | Go to cell end |
| `Ctrl + ←/→` | Jump by word |

---

## 7. Display & Rich Output

### Display Functions

```python
from IPython.display import display, HTML, Markdown, Image, Audio, Video

# Display multiple outputs from same cell
x = 10
y = 20
display(x)
display(y)

# Display HTML
display(HTML("<h1>Hello</h1>"))
display(HTML("<span style='color:red'>Red text</span>"))

# Display Markdown
display(Markdown("**Bold** and *italic*"))
display(Markdown("# Header\n- Item 1\n- Item 2"))

# Display Image
display(Image(url="https://example.com/image.png"))
display(Image(filename="local_image.png"))
display(Image(data=image_bytes))
```

### Pretty Printing

```python
from pprint import pprint

# Standard print (hard to read)
print({"a": [1, 2, 3], "b": {"c": 4, "d": 5}})

# Pretty print (formatted)
pprint({"a": [1, 2, 3], "b": {"c": 4, "d": 5}})
```

### Display DataFrames

```python
import pandas as pd
from IPython.display import display

df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Multiple dataframes in one cell
display(df.head())
display(df.describe())

# Control display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
```

### Progress Bars

```python
from tqdm.notebook import tqdm
import time

# Progress bar for loops
for i in tqdm(range(100)):
    time.sleep(0.01)

# With description
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)
```

---

## 8. Debugging

### Post-Mortem Debugging

```python
# Enable automatic debugger on exception
%pdb on

# Or manually after error
%debug         # Enter debugger at last exception

# Debug specific function call
%run -d script.py    # Run with debugger
```

### Debugger Commands (pdb)

| Command | Action |
|---------|--------|
| `h` | Help |
| `n` | Next line (step over) |
| `s` | Step into function |
| `c` | Continue execution |
| `q` | Quit debugger |
| `p var` | Print variable |
| `pp var` | Pretty print variable |
| `l` | List source code |
| `w` | Show call stack |
| `u` / `d` | Move up/down call stack |
| `b N` | Set breakpoint at line N |

### Set Breakpoints in Code

```python
def my_function(x):
    result = x * 2
    breakpoint()       # Python 3.7+ built-in
    # or: import pdb; pdb.set_trace()
    return result + 1
```

---

## 9. Configuration & Profiles

### IPython Configuration

```bash
# Generate config file
ipython profile create

# Config location
~/.ipython/profile_default/ipython_config.py
```

### Common Configuration Options

```python
# In ipython_config.py
c.InteractiveShell.ast_node_interactivity = "all"  # Show all outputs
c.InteractiveShell.automagic = True                # Use magics without %
c.TerminalInteractiveShell.confirm_exit = False    # Don't ask on exit
```

### Jupyter Configuration

```bash
# Generate config
jupyter notebook --generate-config

# Config location
~/.jupyter/jupyter_notebook_config.py
```

### Extensions

```python
# Load extensions
%load_ext autoreload     # Auto-reload modules
%autoreload 2            # Reload all modules before execution

# Useful for development - no need to restart kernel
```

---

## 10. Common Traps

### ⚠️ Trap 1: Cell Execution Order

```python
# Cells can be run in any order!
# This causes hidden state bugs

# Cell 1 (run second)
x = x + 1    # x is 11

# Cell 2 (run first)  
x = 10       # x is 10

# Always restart kernel and run all to ensure reproducibility
# Kernel → Restart & Run All
```

### ⚠️ Trap 2: Shell `cd` Doesn't Persist

```python
!cd /some/path    # Changes directory in subshell only
!pwd              # Still in original directory!

# Use magic command instead
%cd /some/path    # ✅ Changes directory persistently
%pwd              # Shows new directory
```

### ⚠️ Trap 3: `%%` Must Be First Line

```python
# ❌ WRONG - won't work
import numpy as np
%%timeit
np.sum(range(1000))

# ✅ CORRECT
%%timeit
import numpy as np
np.sum(range(1000))
```

### ⚠️ Trap 4: Kernel State After Variable Deletion

```python
# Deleting cell doesn't delete variable!
x = 100  # Cell executed then deleted

# x still exists in memory until kernel restart
print(x)  # 100 (still there!)

# Fix: %reset or restart kernel
```

### ⚠️ Trap 5: Automagic Confusion

```python
# With automagic ON (default), % is optional
timeit sum(range(100))     # Works
run script.py              # Works

# But this can conflict with Python names!
time = 5                   # Now 'time' is a variable
time sum(range(100))       # ❌ Error! Tries to use variable 'time'
%time sum(range(100))      # ✅ Explicit magic works
```

---

## 11. Quick Reference

### Getting Help
```python
obj?               # Quick help
obj??              # Source code
obj.<TAB>          # Auto-complete
*pattern?          # Wildcard search
```

### Essential Magics
```python
%run script.py     # Run script
%timeit code       # Benchmark line
%%timeit           # Benchmark cell
%time code         # Time once
%pwd               # Current directory
%cd path           # Change directory
%who / %whos       # List variables
%reset             # Clear namespace
%history           # Command history
%debug             # Post-mortem debug
```

### Shell Commands
```python
!ls                # List files
!pwd               # Working directory
!pip install pkg   # Install package
files = !ls        # Capture output
!cmd {var}         # Use Python variable
```

### History
```python
_                  # Last output
__                 # Second-to-last output
_N                 # Output N
In[N]              # Input N
%history           # All history
```

### Cell Output
```python
display(obj)       # Show object
result;            # Suppress output (semicolon)
```

### Keyboard Shortcuts
```
Shift+Enter        # Run cell, next
Esc+A              # Insert above
Esc+B              # Insert below
Esc+M              # To markdown
Esc+Y              # To code
Esc+D,D            # Delete cell
```

---

**📝 Pro Tips:**
- Use `Shift+Tab` inside function parentheses for instant documentation
- Add `;` at end of matplotlib plots to suppress text output
- Use `%store` to pass variables between notebooks
- Always "Restart & Run All" before sharing notebooks
- Use `%%capture` to suppress all output from a cell

---

