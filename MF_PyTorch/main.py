import data
from MF import MF


def main():
    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data("validation")

    mf = MF(data_splitter.n_user, data_splitter.n_item, 20, 0.001, 0)
    print(mf)

if __name__ == "__main__":
    main()