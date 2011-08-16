import numpy as np 
import scipy
import scipy.signal 
import scipy.stats

def avg(x,y):
    return (x+y)/2
            
def replace_inf(x,y):
    inf = np.isinf(x)
    if np.any(inf):
        x = x.copy()
        x[inf] = y 
    return x
    
def replace_nan(x,y):
    nan = np.isnan(x)
    if np.any(nan):
        x = x.copy()
        x[nan] = y 
    return x


def replace_inf_or_nan(x,y):
    return replace_inf(replace_nan(x,y), y)

# ratio between two quantities, in the presence of zeros and weird extrema
# if a ratio is bad (too small, too big, NaN, or inf) set it to 1.0 
def safe_div(x,y):
    x_zero = x == 0
    y_zero = y == 0
    if np.any(x_zero) or np.any(y_zero):
        x = x.copy()
        y = y.copy()
        both_zero = np.logical_and(x_zero, y_zero)
        x[both_zero] = 1.0
        y[both_zero] = 1.0
        y[y_zero] = x[y_zero] 
    z = x / y
    return clean(z)


def add(x,y):
    return x + y 
    
def sub(x,y):
    return x - y
    
def mult(x,y):
    return x * y 
    
def div(x,y):
    return x / y 

def gt(x,y):
    return x > y
    
def gte(x,y):
    return x >= y
    
def lt(x,y):
    return x < y
    
def lte(x,y):
    return x <= y
    
def eq(x,y):
    return x == y 

def select_bigger_abs(x,y):
    return np.where(np.abs(x) > np.abs(y), x, y)
    
def select_smaller_abs(x,y):
    return np.where(np.abs(x) < np.abs(y), x, y)

def when(x,y):
    return x[y]
    
def medfilt(x, winsize):
    return scipy.signal.medfilt(x, winsize)

binops = { 
    '+': add, #np.add,
    '-': sub, #np.subtract, 
    '*': mult, #np.multiply, 
    '%': div, #np.divide,
    '>': gt,
    '>=': gte,
    '<': lt, 
    '<=': lte, 
    '=': eq,
    'mod': np.mod,
    'avg': avg, 
    'min': np.minimum, 
    'max': np.maximum, 
    'replace_inf': replace_inf,
    'replace_nan': replace_nan, 
    'replace_inf_or_nan': replace_inf_or_nan, 
    'safe_div': safe_div,
    'select_bigger_abs': select_bigger_abs,
    'select_smaller_abs': select_smaller_abs, 
    'when': when, 
    'medfilt': medfilt
}


def clean(x):
    top = scipy.stats.scoreatpercentile(x, 99)
    bottom = scipy.stats.scoreatpercentile(x, 1)    
    outliers = np.abs(x) > 2 * top 
    if np.any(outliers):
        x = x.copy()
        x[outliers & (x < 0)] = bottom
        x[outliers & (x > 0)] = top 
    return x

def diff(x):
    return np.concatenate([[0], np.diff(x)])
    
unops = { 
    'diff': diff, 
    'log': np.log, 
    'log10': np.log10, 
    'log2': np.log2, 
    'sin': np.sin, 
    'cos': np.cos, 
    'tan': np.tan, 
    'std': np.std, 
    'mean': np.mean, 
    'abs': np.abs, 
    'clean': clean,
    
}
def tokenize(s):
    # remove quotes and strip whitespace
    s = s.replace('"', '')
    s = s.strip()
    
    tokens = []
    curr = ''
    special = ['+', '-', '%', '*', '(', ')'] 
    for c in s:
        is_special = c in special 
        if c == ' ' or c == '\t' or is_special:
            if len(curr) > 0:
                tokens.append(curr)
                curr = ''
            if is_special: tokens.append(c)
        else:
            curr = curr + c 
    if curr != '': tokens.append(curr)
    return tokens 


def mk_const_fn(const):
    def fn(env):
        return const
    return fn 
    
def mk_var_fn(name):
    def fn(env):
        return env[name]
    return fn 
    
def mk_unop_fn(unop_name, arg): 
    unop = unops[unop_name] 
    def fn(env):
        arg_val = arg(env)
        return unop(arg_val)
    return fn 
    
def mk_binop_fn(binop_name, left, right): 
    binop = binops[binop_name] 
    def fn(env):
        left_val = left(env)
        right_val = right(env)
        return binop(left_val, right_val)
    return fn 
    
def compile_tokens(tokens):
    
    # a stack of 0-ary function used as arguments to the mk_xyz functions above
    curr_value_stack = [] 
    curr_waiting_binops = []
    
    old_value_stacks = []
    old_waiting_binops = [] 
    
    # reversed since we evaluate right to left 
    for token in reversed(tokens): 
        if token in unops:
            arg = curr_value_stack.pop()
            future_unop_result = mk_unop_fn(token, arg)
            curr_value_stack.append(future_unop_result)
        elif token in binops:
            curr_waiting_binops.append(token)
            
        elif token == ')':
            old_value_stacks.append(curr_value_stack)
            old_waiting_binops.append(curr_waiting_binops)
            curr_value_stack = []
            curr_waiting_binops = [] 
            
        elif token == '(':
            v = curr_value_stack.pop()
            curr_value_stack = old_value_stacks.pop()
            curr_waiting_binops = old_waiting_binops.pop()
            if len(curr_waiting_binops) > 0:
                binop_name = curr_waiting_binops.pop()
                rightArg = curr_value_stack.pop()
                future_result = mk_binop_fn(binop_name, v, rightArg)
                curr_value_stack.append(future_result) 
            else:
                curr_value_stack.append(v) 
        else:
            try:
                const = float(token)
                arg = mk_const_fn(const) 
            except: 
                arg = mk_var_fn(token)
            if len(curr_waiting_binops) > 0:
                binop_name = curr_waiting_binops.pop()
                rightArg = curr_value_stack.pop() 
                future_result = mk_binop_fn(binop_name, arg, rightArg)
                curr_value_stack.append(future_result)
            else:
                curr_value_stack.append(arg)
    return curr_value_stack.pop() 
    
# returns a function which takes an environment dictionary and returns the 
# value of the evaluated expression  
def compile_expr(expr):
    tokens = tokenize(expr)
    return compile_tokens(tokens)

