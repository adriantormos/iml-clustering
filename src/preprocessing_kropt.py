from src.auxiliary import load_arff
import pandas as pd

if __name__ == '__main__':
    data, metadata = load_arff('kropt')
    data = pd.DataFrame(data)

    for col in data.columns:
        print(data[col].dtype)
    pass