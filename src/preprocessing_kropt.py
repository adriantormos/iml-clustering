from src.auxiliary import load_arff
import pandas as pd
from matplotlib import pyplot as plt


krops_category_mapping = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8
}


def transform_krops_col_to_numeric(column, column_name: str):
    if 'row' in column_name:
        return [int(x.decode('utf-8')) for x in column]
    return [krops_category_mapping[x.decode('utf-8')] for x in column]


if __name__ == '__main__':
    data, metadata = load_arff('kropt')
    data = pd.DataFrame(data)

    # Transform all columns to numeric values
    for col in data.columns:
        if col != 'game':
            data[col] = transform_krops_col_to_numeric(data[col], col)
            # plt.figure()
            # plt.hist(data[col], bins=8)
            # plt.title(col)
            # plt.show()

    # Replicate rows to balance the dataset
    print(data[data['game'] == b'draw'])
    data = data.append([data[data['game'] == b'zero']] * 75, ignore_index=True)
    data = data.append([data[data['game'] == b'one']] * 30, ignore_index=True)
    data = data.append([data[data['game'] == b'two']] * 10, ignore_index=True)
    data = data.append([data[data['game'] == b'three']] * 30, ignore_index=True)
    data = data.append([data[data['game'] == b'four']] * 12, ignore_index=True)
    data = data.append([data[data['game'] == b'five']] * 6, ignore_index=True)
    data = data.append([data[data['game'] == b'six']] * 6, ignore_index=True)
    data = data.append([data[data['game'] == b'seven']] * 5, ignore_index=True)
    data = data.append([data[data['game'] == b'eight']] * 2, ignore_index=True)
    data = data.append([data[data['game'] == b'nine']], ignore_index=True)
    data = data.append([data[data['game'] == b'ten']], ignore_index=True)
    data = data.append([data[data['game'] == b'fifteen']], ignore_index=True)
    data = data.append([data[data['game'] == b'sixteen']] * 8, ignore_index=True)

    print(data['game'].value_counts())

    # Data visualization

    wk = []
    wr = []
    bk = []

    for result in data['game'].unique():
        result_df = data[data['game'] == result]
        plt.figure()
        plt.title('Average positions in ' + result.decode('utf-8'))
        plt.plot(result_df['white_king_col'].mean(), result_df['white_king_row'].mean(), 'yo', label='White king')
        plt.plot(result_df['white_rook_col'].mean(), result_df['white_rook_row'].mean(), 'go', label='White rook')
        plt.plot(result_df['black_king_col'].mean(), result_df['black_king_col'].mean(), 'bo', label='Black king')
        plt.xlim([0, 9])
        plt.ylim([0, 9])
        plt.legend()
        plt.show()
        wk.append((result_df['white_king_col'].mean(), result_df['white_king_row'].mean()))
        wr.append((result_df['white_rook_col'].mean(), result_df['white_rook_row'].mean()))
        bk.append((result_df['black_king_col'].mean(), result_df['black_king_col'].mean()))

    plt.figure()
    plt.title('Piece evolution')
    plt.plot([x[0] for x in wk], [x[1] for x in wk], 'y-', label='White king')
    plt.plot([x[0] for x in wr], [x[1] for x in wr], 'g-', label='White rook')
    plt.plot([x[0] for x in bk], [x[1] for x in bk], 'b-', label='Black king')
    plt.xlim([0, 9])
    plt.ylim([0, 9])
    plt.legend()
    plt.show()

    pass