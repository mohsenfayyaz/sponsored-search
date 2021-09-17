import pickle

import pandas as pd
from tqdm.auto import tqdm
import random
from transformers import AutoTokenizer
import datasets
from sklearn.model_selection import train_test_split
from src.Utils import Utils


class DatasetHandler:
    def __init__(self, file_address, batch_size=32, frac=1):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

        print("Reading input file...")
        self.df = pd.read_csv(file_address) if "csv" in file_address else pd.read_pickle(file_address)
        self.df['queryText'] = self.df['queryText'].astype(str)
        self.package_to_id, self.id_to_package = self.label_packages(self.df["packageName"])

        print("Creating dataset...")
        self.dataset = self.df_to_dataset(self.df, fraction=frac)
        self.dataset = self.dataset.map(Utils.tokenize_query,
                                        fn_kwargs={"tokenizer": self.tokenizer},
                                        batched=True,
                                        batch_size=batch_size,
                                        num_proc=None)
        self.dataset = self.dataset.map(Utils.tokenize_ad,
                                        fn_kwargs={"package_to_id": self.package_to_id},
                                        batched=False,
                                        num_proc=None)
        print("Done")

    def get_tokenizer(self):
        return self.tokenizer

    def get_dataset(self):
        return self.dataset

    def get_query_vocab_size(self):
        return len(self.tokenizer.get_vocab())

    def get_ad_vocab_size(self):
        return len(self.package_to_id)

    def save_id_to_package(self, output_address="id_to_package.pkl"):
        with open(output_address, 'wb') as f:
            pickle.dump(self.id_to_package, f, pickle.HIGHEST_PROTOCOL)

    def df_to_dataset(self, data_df, fraction, shuffle=True, random_state=0, test_size=0.2):
        if shuffle:
            data_df = data_df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        else:
            data_df = data_df.sample(frac=fraction, random_state=random_state).sort_index().reset_index(drop=True)
        self.df = data_df
        train_df, test_df = train_test_split(data_df, test_size=test_size)
        self.dataset = datasets.DatasetDict()
        self.dataset["train"] = datasets.Dataset.from_pandas(train_df)
        self.dataset["test"] = datasets.Dataset.from_pandas(test_df)
        return self.dataset

    def label_packages(self, packages, start_id=0):
        packages_set = set(packages.unique())
        package_to_id = dict()
        id_to_package = dict()
        id_counter = start_id
        for package in tqdm(packages_set, desc="Labeling Packages"):
            package_to_id[package] = id_counter
            id_to_package[id_counter] = package
            id_counter += 1
        return package_to_id, id_to_package
