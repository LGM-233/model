from datasets import load_dataset

ds = load_dataset("mwritescode/slither-audited-smart-contracts", "big-multilabel",trust_remote_code=True)
