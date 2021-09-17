import pickle

import torch
from transformers import AutoTokenizer
from src.Trainer.QueryAdCoordinator import QueryAdCoordinator
from src.Utils import Utils


class Predictor:
    def __init__(self, ad_reprs_address="representations/ad_reprs.pt",
                 id_to_package_address="representations/id_to_package.pkl",
                 query_ad_coordinator_checkpoint="representations/QueryAdCoordinator_checkpoint.pt",
                 device="cpu"):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        print("Loading ad representations...")
        self.ad_reprs = torch.load(ad_reprs_address)
        print("Loading ad id to package...")
        with open(id_to_package_address, 'rb') as fp:
            self.ad_id_to_package = pickle.load(fp)
        print("Loading QueryAdCoordinator...")
        self.query_ad_coordinator = torch.load(query_ad_coordinator_checkpoint)
        self.device = device
        self.query_ad_coordinator.to(self.device)

    def predict_from_query(self, query: str):
        tokenized_query = self.tokenizer([query], padding=True)
        input_ids = torch.tensor(tokenized_query["input_ids"]).to(self.device)
        attention_mask = torch.tensor(tokenized_query["attention_mask"]).to(self.device)
        ad_repr = self.query_ad_coordinator.build_query_representation(input_ids, attention_mask)
        print(ad_repr.shape)
        print(ad_repr)
