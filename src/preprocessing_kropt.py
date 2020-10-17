krops_category_mapping = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8
}


def transform_krops_col_to_numeric(column, column_name: str):
    if 'row' in column_name:
        return [int(x.decode('utf-8')) for x in column]
    return [krops_category_mapping[x.decode('utf-8')] for x in column]
