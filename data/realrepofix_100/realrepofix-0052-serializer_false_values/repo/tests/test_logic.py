from samplepkg.logic import serialize_user


def test_serialize_user_keeps_false_values():
    assert serialize_user({'name': 'Ada', 'active': False, 'age': 0}) == {'name': 'Ada', 'active': False, 'age': 0}
