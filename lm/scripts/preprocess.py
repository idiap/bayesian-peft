#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Haolin Chen <haolin.chen@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
from datasets import load_dataset, Dataset


# https://huggingface.co/datasets/monology/pile-uncopyrighted
val_path = "data/val.json"
test_path = "data/test.json"

full_set = load_dataset("json", data_files=[val_path, test_path], split="train")
full_set.shuffle(2023)
full_set.to_json("data/pile.json")

pile_20000 = Dataset.from_dict(full_set[:20000])
pile_20000.to_json("data/pile_20000.json")
pile_2000 = Dataset.from_dict(full_set[-2000:])
pile_2000.to_json("data/pile_2000.json")

metas = full_set["meta"]
pile_set_names = []
for meta in metas:
    pile_set_names.append(meta["pile_set_name"])
unique_pile_set_names = list(set(pile_set_names))
print(unique_pile_set_names)
# ['FreeLaw', 'PubMed Abstracts', 'USPTO Backgrounds', 'Pile-CC', 'Wikipedia (en)', 
# 'Gutenberg (PG-19)', 'Enron Emails', 'PubMed Central', 'DM Mathematics', 'NIH ExPorter', 
# 'ArXiv', 'HackerNews', 'EuroParl', 'Github', 'PhilPapers', 
# 'Ubuntu IRC', 'StackExchange']

opt_subset_names = ['Pile-CC', 'DM Mathematics', 'Gutenberg (PG-19)', 'HackerNews', 'USPTO Backgrounds', 'Wikipedia (en)']
opt_full_set = full_set.filter(lambda example: example["meta"]["pile_set_name"] in opt_subset_names)
print(len(opt_full_set))

opt_full_set.to_json("data/opt_pile.json")

opt_20000 = Dataset.from_dict(opt_full_set[:20000])
opt_20000.to_json("data/opt_pile_20000_train.json")
opt_2000_test = Dataset.from_dict(opt_full_set[-2000:])
opt_2000_test.to_json("data/opt_pile_2000_test.json")
opt_2000_train = Dataset.from_dict(opt_full_set[:2000])
opt_2000_train.to_json("data/opt_pile_2000_train.json")
opt_200 = Dataset.from_dict(opt_full_set[:200])
opt_200.to_json("data/opt_pile_200_train.json")
opt_20 = Dataset.from_dict(opt_full_set[60:80])
opt_20.to_json("data/opt_pile_20_train.json")
