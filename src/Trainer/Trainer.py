import torch
from tqdm.auto import tqdm
from src.Trainer.DatasetHandler import DatasetHandler
from src.Trainer.QueryAdCoordinator import QueryAdCoordinator
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, dataset_handler: DatasetHandler, embedding_dim, learning_rate=5e-4, device="cuda"):
        self.history = {
            "loss": {
                "train": [],
                "test": []
            }
        }
        self.dataset_handler = dataset_handler
        self.device = device
        self.query_ad_coordinator = QueryAdCoordinator(query_vocab_size=self.dataset_handler.get_query_vocab_size(),
                                                       ad_vocab_size=self.dataset_handler.get_ad_vocab_size(),
                                                       embedding_dim=embedding_dim,
                                                       learning_rate=learning_rate)
        self.batch_size = 32

    def train(self, batch_size=32, epochs=3):
        self.batch_size = batch_size
        tokenized_dataset_train = self.dataset_handler.get_dataset()["train"]
        tokenized_dataset_test = self.dataset_handler.get_dataset()["test"]
        train_dataset_len = len(tokenized_dataset_train)
        test_dataset_len = len(tokenized_dataset_test)
        print(f"Train on {train_dataset_len} samples, test on {test_dataset_len} samples")
        self.query_ad_coordinator.to(self.device)

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
                query_repr, ad_repr = self.query_ad_coordinator(*batch_inputs)

                loss = self.query_ad_coordinator.loss(query_repr.to(self.device), ad_repr.to(self.device),
                                                      labels.float().to(self.device))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.query_ad_coordinator.parameters(), 5.0)
                self.query_ad_coordinator.optimizer.step()

                running_loss += loss.item()
                steps += 1

            self.update_history(train_loss=running_loss / steps)
            self.plot_history()

    def prepare_batch(self, tokenized_dataset, start_idx, end_idx):
        batch = tokenized_dataset[start_idx: end_idx]
        x1 = torch.tensor(batch["input_ids"]).to(self.device)
        x2 = torch.tensor(batch["package_ids"]).to(self.device)
        attention_mask1 = torch.tensor(batch["attention_mask"]).to(self.device)
        attention_mask2 = torch.ones(len(x2), 1, device=self.device)
        labels = torch.tensor(batch["similar"]).to(self.device)
        return [x1, x2, attention_mask1, attention_mask2], labels

    def update_history(self, train_loss=0.0):
        test_loss = self.calc_loss(self.dataset_handler.get_dataset()["test"], desc="Test Loss")
        self.history["loss"]["train"].append(train_loss)
        self.history["loss"]["test"].append(test_loss)

    def plot_history(self):
        plt.title("Loss History")
        loss_history = self.history["loss"]
        x = range(len(loss_history["train"]))
        plt.plot(x, loss_history["train"])
        plt.plot(x, loss_history["test"])
        plt.legend(['Train', 'Test'], loc='lower left')
        plt.show()

    def calc_loss(self, tokenized_dataset, desc=""):
        self.query_ad_coordinator.eval()
        with torch.no_grad():
            running_loss = 0
            dataset_len = len(tokenized_dataset["input_ids"])
            steps = 0
            for i in tqdm(range(0, dataset_len, self.batch_size), desc=desc):
                step = self.batch_size
                if i + self.batch_size > dataset_len:
                    step = dataset_len - i

                batch_inputs, labels = self.prepare_batch(tokenized_dataset, i, i + step)
                query_repr, ad_repr = self.query_ad_coordinator(*batch_inputs)

                loss = self.query_ad_coordinator.loss(query_repr.to(self.device), ad_repr.to(self.device),
                                                      labels.float().to(self.device))

                running_loss += loss.item()
                steps += 1

        return running_loss / steps

    def save_model(self, save_address="representations/QueryAdCoordinator_checkpoint.pt"):
        self.query_ad_coordinator.save_model(save_address)

    def load_model(self):
        self.query_ad_coordinator.load_model()

    def save_all_representations(self, ad_reprs_address="representations/ad_reprs.pt",
                                 vocab_reprs_address="representations/vocab_reprs.pt",
                                 id_to_package_address="representations/ad_id_to_package.pkl"):
        self.dataset_handler.save_id_to_package(id_to_package_address)
        self.save_ad_reprs(ad_reprs_address)
        self.save_vocab_reprs(vocab_reprs_address)

    def save_ad_reprs(self, path, device="cpu"):
        self.query_ad_coordinator.to(device)
        ad_representations = []
        for ad_id in tqdm(sorted(self.dataset_handler.id_to_package), desc="Building ad representations"):
            ad_id_tensor = torch.tensor([[ad_id]]).to(device)
            attention_mask = torch.ones(1, 1, device=device)
            ad_repr = self.query_ad_coordinator.build_ad_representation(ad_id_tensor, attention_mask)
            ad_representations.append(ad_repr[0])

        ad_representations = torch.stack(ad_representations)
        print(ad_representations.shape)
        torch.save(ad_representations, path)
        print(f"Saved ad representations at {path}")
        self.query_ad_coordinator.to(self.device)

    def save_vocab_reprs(self, path, device="cpu"):
        self.query_ad_coordinator.to(device)
        vocab_representations = dict()
        for word, word_id in tqdm(self.dataset_handler.tokenizer.get_vocab().items(),
                                  desc="Building vocab representations"):
            word_id_tensor = torch.tensor([[word_id]]).to(device)
            attention_mask = torch.ones(1, 1, device=device)
            word_repr = self.query_ad_coordinator.build_query_representation(word_id_tensor, attention_mask)
            vocab_representations[word_id] = word_repr
        print(len(vocab_representations))
        torch.save(vocab_representations, path)
        print(f"Saved vocab representations at {path}")
        self.query_ad_coordinator.to(self.device)
