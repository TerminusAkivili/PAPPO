from samplepkg.logic import parse_csv_line


def test_parse_csv_line_handles_quoted_comma():
    assert parse_csv_line('a,"b,c",d') == ['a', 'b,c', 'd']
