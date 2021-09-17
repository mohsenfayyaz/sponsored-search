import pickle

import torch
from flask import jsonify
from transformers import AutoTokenizer
from src.Trainer.QueryAdCoordinator import QueryAdCoordinator
from src.Utils import Utils


class Predictor:
    def __init__(self, ad_reprs_address="representations/ad_reprs.pt",
                 id_to_package_address="representations/id_to_package.pkl",
                 query_ad_coordinator_checkpoint="representations/QueryAdCoordinator_checkpoint.pt",
                 device="cpu"):
        self.device = device
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        print("Loading ad representations...")
        self.ad_reprs = torch.load(ad_reprs_address)
        print("Loading ad id to package...")
        with open(id_to_package_address, 'rb') as fp:
            self.ad_id_to_package = pickle.load(fp)
        print("Loading QueryAdCoordinator...")
        self.query_ad_coordinator = torch.load(query_ad_coordinator_checkpoint)
        self.query_ad_coordinator.to(self.device)

    def predict_from_query(self, query: str):
        tokenized_query = self.tokenizer([query], padding=True)
        input_ids = torch.tensor(tokenized_query["input_ids"]).to(self.device)
        attention_mask = torch.tensor(tokenized_query["attention_mask"]).to(self.device)
        query_repr = self.query_ad_coordinator.build_query_representation(input_ids, attention_mask)
        # print(query_repr.shape)
        # print(query_repr)
        knn_values, knn_ids = self.knn(query_repr, self.ad_reprs.to(self.device), k=10)
        # print(self.ad_id_to_package)
        # print(query)
        packages = []
        for value, ad_id in zip(knn_values, knn_ids):
            packages.append(self.ad_id_to_package[int(ad_id)])
            # print(f"dist: {value:.2f}", packages[-1])
        return packages

    def knn(self, query_repr: torch.tensor, ads_reprs: torch.tensor, k=5):
        # dist = torch.norm(query_repr - ads_reprs, dim=1, p=None)
        dist = torch.nn.functional.cosine_similarity(query_repr, ads_reprs, dim=1)
        knn = dist.topk(k, largest=True)
        print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
        return knn.values, knn.indices
