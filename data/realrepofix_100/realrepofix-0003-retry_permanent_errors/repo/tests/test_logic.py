from samplepkg.logic import should_retry


def test_should_retry_skips_permanent_errors():
    assert should_retry('TimeoutError') is True
    assert should_retry('ValidationError') is False
