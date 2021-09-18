import random
from tqdm.auto import tqdm
import pandas as pd


class Preprocessor:
    @staticmethod
    def create_negative_sampled_dataset(input_address, output_address="../data/neg_sampled_data.pkl",
                                        neg_to_pos_ratio: int = 5, seed=0, pos_class=1, neg_class=-1):
        """
        :param input_address: input csv address
        :param output_address: output pkl address
        :param neg_to_pos_ratio: how many negative samples to be generated for each positive sample
        :param seed: random seed for selecting negative classes
        :param neg_class: Value for positive classes in similar column
        :param pos_class: Value for positive classes in negative column
        :return: dataframe of the new dataset
        """
        print(f"Reading input file {input_address}")
        df = pd.read_csv(input_address) if "csv" in input_address else pd.read_pickle(input_address)

        df["similar"] = pos_class
        random.seed(seed)
        negative_rows = []
        random_choices = list(set(df["packageName"].unique()))
        for query_text, package_name in tqdm(zip(df["queryText"], df["packageName"]),
                                             total=len(df["queryText"])):
            for _ in range(neg_to_pos_ratio):
                random_package = random.choice(random_choices)
                while random_package == package_name:
                    random_package = random.choice(random_choices)
                negative_rows.append({'queryText': query_text, "packageName": random_package, "similar": neg_class})
        neg_df = pd.DataFrame(negative_rows)
        new_df = pd.concat([df, neg_df], ignore_index=True)
        new_df.to_pickle(output_address)
        print(f"File created at {output_address}")
        return new_df
