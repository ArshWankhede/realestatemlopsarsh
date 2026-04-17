def test_math_check():
    """A simple 'Smoke Test' to ensure CI is working"""
    assert 1 + 1 == 2

def test_environment_libs():
    """Ensure key libraries are installed"""
    import pandas
    import sklearn
    assert True