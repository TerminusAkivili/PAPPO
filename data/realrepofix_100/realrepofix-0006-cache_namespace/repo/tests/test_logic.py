from samplepkg.logic import cache_key


def test_cache_key_includes_namespace():
    assert cache_key('users', '42') != cache_key('orders', '42')
    assert cache_key('users', '42') == 'users:42'
