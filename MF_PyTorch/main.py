import data


def main():
    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data("validation")


if __name__ == "__main__":
    main()