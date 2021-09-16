import torch
from tqdm.auto import tqdm
from src.DatasetHandler import DatasetHandler
from src.QueryAdCoordinator import QueryAdCoordinator


class Trainer:
    def __init__(self, embedding_dim, dataset_handler: DatasetHandler, device="cuda"):
        self.dataset_handler = dataset_handler
        self.device = device
        self.query_ad_coordinator = QueryAdCoordinator(self.dataset_handler.get_vocab_size(), embedding_dim)

    def train(self, batch_size=32, epochs=3):
        tokenized_dataset_train = self.dataset_handler.get_dataset()["train"]
        tokenized_dataset_test = self.dataset_handler.get_dataset()["test"]
        train_dataset_len = len(tokenized_dataset_train)
        test_dataset_len = len(tokenized_dataset_test)
        print(f"Train on {train_dataset_len} samples, test on {test_dataset_len} samples")
        print("A")
        self.query_ad_coordinator.to(self.device)

        # self.query_ad_coordinator(torch.tensor([[1], [0, 0, 0, 0]]), torch.tensor([[1, 2, 3, 3], [9, 9, 9, 9]]))
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            print("----------------\n")
            self.query_ad_coordinator.train()
            for i in tqdm(range(0, train_dataset_len, batch_size), desc=f"[Epoch {epoch + 1}/{epochs}]"):
                step = batch_size
                if i + batch_size > train_dataset_len:
                    step = train_dataset_len - i

                batch_inputs, labels = self.prepare_batch(tokenized_dataset_train, i, i + step)
                outputs = self.query_ad_coordinator(*batch_inputs)

                # loss = self.edge_probe_model.training_criterion(outputs.to(self.device), labels.float().to(self.device))
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.edge_probe_model.parameters(), 5.0)
                # self.edge_probe_model.optimizer.step()
                #
                # running_loss += loss.item()
                # steps += 1

    def prepare_batch(self, tokenized_dataset, start_idx, end_idx):
        batch = tokenized_dataset[start_idx: end_idx]
        x1 = torch.tensor(batch["input_ids"]).to(self.device)
        x2 = torch.tensor(batch["package_ids"]).to(self.device)
        attention_mask1 = torch.tensor(batch["attention_mask"]).to(self.device)
        attention_mask2 = torch.tensor(batch["attention_mask"]).to(self.device)
        labels = torch.tensor(batch["similar"]).to(self.device)
        return [x1, x2, attention_mask1, attention_mask2], labels
