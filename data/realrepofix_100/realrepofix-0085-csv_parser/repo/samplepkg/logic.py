import csv
from io import StringIO

def parse_csv_line(line):
    reader = csv.reader(StringIO(line))
    return next(reader)
