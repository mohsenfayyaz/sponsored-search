class Utils:
    @staticmethod
    def tokenize_query(examples, **fn_kwargs):
        tokenizer = fn_kwargs["tokenizer"]
        return tokenizer(examples['queryText'], padding=True)

    @staticmethod
    def tokenize_ad(examples, **fn_kwargs):
        package_to_id = fn_kwargs["package_to_id"]
        examples["package_ids"] = [package_to_id[examples["packageName"]]]
        return examples
