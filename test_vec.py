import vec
import numpy as np

def test_compiler_simple():
    code = vec.compile_expr('3+4')
    assert code({}) == 7 
    
def test_compiler_env():
    env = {'x':3, 'y':4}
    code = vec.compile_expr('x+y')
    res1 = code(env)
    print "Expected 7, got", res1 
    assert res1 == 7
    env['x'] = 4
    res2 = code(env)
    print "Expected 8, got", res2 
    assert res2 == 8

def test_compiler_nested():
    code = vec.compile_expr('(10+4)%2')
    res = code({})
    print "Received: ", res
    assert res == 7 
    
def test_compile_array(): 
    code = vec.compile_expr('(log2 x) + (log2 y)')
    env = {'x': np.array([2, 4]), 'y': np.array([4, 8])}
    res = code(env)
    expected = np.array([3,5])
    print "Expected:", expected, "Received:", res 
    assert np.all(res == expected)
