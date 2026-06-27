from samplepkg.logic import is_expired


def test_is_expired_at_boundary():
    assert is_expired(10, 10) is True
    assert is_expired(9, 10) is False
