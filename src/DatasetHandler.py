import pandas as pd
from tqdm.auto import tqdm
import random
from transformers import AutoTokenizer
import datasets
from sklearn.model_selection import train_test_split


class DatasetHandler:
    def __init__(self, file_address):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        print("Reading input file...")
        self.df = pd.read_csv(file_address) if "csv" in file_address else pd.read_pickle(file_address)
        self.package_to_id, self.id_to_package = self.label_packages(self.df["packageName"],
                                                                     start_id=len(self.tokenizer.get_vocab()))
        print("Creating dataset...")
        self.dataset = self.df_to_dataset(self.df)
        print("Done")

    def df_to_dataset(self, data_df, shuffle=True, fraction=0.1, random_state=0, test_size=0.2):
        if shuffle:
            data_df = data_df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        else:
            data_df = data_df.sample(frac=fraction, random_state=random_state).sort_index().reset_index(drop=True)
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

    def save_negative_sampled_dataset(self, neg_to_pos_ratio: int = 4, output_address="../data/neg_sampled_data.pkl",
                                      seed=0):
        self.df["similar"] = 1
        random.seed(seed)
        negative_rows = []
        random_choices = list(self.package_to_id.keys())
        for query_text, package_name in tqdm(zip(self.df["queryText"], self.df["packageName"]),
                                             total=len(self.df["queryText"])):
            for _ in range(neg_to_pos_ratio):
                random_package = random.choice(random_choices)
                while random_package == package_name:
                    random_package = random.choice(random_choices)
                negative_rows.append({'queryText': query_text, "packageName": random_package, "similar": 0})
        neg_df = pd.DataFrame(negative_rows)
        new_df = pd.concat([self.df, neg_df], ignore_index=True)
        new_df.to_pickle(output_address)
        return new_df
