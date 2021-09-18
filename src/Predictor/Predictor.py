import pickle
import torch
from transformers import AutoTokenizer


class Predictor:
    def __init__(self, ad_reprs_address="representations/ad_reprs.pt",
                 vocab_reprs_address="representations/vocab_reprs.pt",
                 id_to_package_address="representations/ad_id_to_package.pkl",
                 device="cpu"):
        self.device = device
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", use_fast=False)
        print("Loading ad representations...")
        self.ad_reprs = torch.load(ad_reprs_address, map_location=torch.device(self.device))
        print("Loading ad id to package...")
        with open(id_to_package_address, 'rb') as fp:
            self.ad_id_to_package = pickle.load(fp)
        print("Loading vocab representations...")
        self.vocab_reprs = torch.load(vocab_reprs_address, map_location=torch.device(self.device))
        self.reprs_shape = self.vocab_reprs[0].shape

    def predict_from_query(self, query: str, full_results=False, k=10):
        query_repr = self.build_query_representation(query)
        knn_values, knn_ids = self.knn(query_repr, self.ad_reprs, k)
        packages = []
        for value, ad_id in zip(knn_values, knn_ids):
            packages.append(self.ad_id_to_package[int(ad_id)])
        if full_results:
            for value, ad_id in zip(knn_values, knn_ids):
                print(f"dist: {value:.5f}", self.ad_id_to_package[int(ad_id)])
            return packages, knn_values, knn_ids, query_repr
        else:
            return packages

    def build_query_representation(self, query):
        tokenized_query = self.tokenizer(query, padding=True)
        input_ids = torch.tensor(tokenized_query["input_ids"])
        query_sum = torch.zeros(self.reprs_shape)
        for word_id in input_ids:
            query_sum += self.vocab_reprs[int(word_id)]
        query_repr = query_sum / len(input_ids)
        return query_repr

    def knn(self, query_repr: torch.tensor, ads_reprs: torch.tensor, k=5):
        # dist = torch.norm(query_repr - ads_reprs, dim=1, p=None)
        dist = torch.nn.functional.cosine_similarity(query_repr, ads_reprs, dim=1)
        knn = dist.topk(k, largest=True)
        # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
        return knn.values, knn.indices
