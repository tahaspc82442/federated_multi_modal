import os
import pickle
import math
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden

def read_split(filepath, path_prefix, caption_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            # Assume caption path is the same as image path, just in the captions folder with .txt extension
            caption_path = impath.replace(path_prefix, caption_prefix).replace('.jpg', '.txt')
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
            else:
                caption = None  # No caption available
            item = Datum(impath=impath, label=int(label), classname=classname, caption=caption)
            out.append(item)
        return out

    print(f"Reading split from {filepath}")
    split = read_json(filepath)
    train = _convert(split["train"])
    val = _convert(split["val"])
    test = _convert(split["test"])

    return train, val, test

def read_and_split_data(image_dir, caption_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
    if ignored is None:
        ignored = []
    print("IMAGE DIR ", image_dir)
    categories = listdir_nohidden(image_dir)
    print("CATEGORIES ", categories)
    categories = [c for c in categories if c not in ignored]
    categories.sort()
    p_tst = 1 - p_trn - p_val
    print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

    all_data = []
    for label, category in enumerate(categories):
        image_category_dir = os.path.join(image_dir, category)
        caption_category_dir = os.path.join(caption_dir, category)
        images = listdir_nohidden(image_category_dir)

        for image_file in images:
            image_path = os.path.join(image_category_dir, image_file)
            caption_file = image_file.replace('.jpg', '.txt')
            caption_path = os.path.join(caption_category_dir, caption_file)

            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
            else:
                caption = None

            item = Datum(impath=image_path, label=label, classname=category, caption=caption)
            all_data.append(item)

    # Shuffle and split the data into train, val, test
    #random.shuffle(all_data)
    num_total = len(all_data)
    num_trn = int(p_trn * num_total)
    num_val = int(p_val * num_total)

    train = all_data[:num_trn]
    val = all_data[num_trn:num_trn + num_val]
    test = all_data[num_trn + num_val:]

    return train, val, test

def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            impath = item.impath
            label = item.label
            classname = item.classname
            caption = item.caption  # Save caption as well
            impath = impath.replace(path_prefix, "")
            if impath.startswith("/"):
                impath = impath[1:]
            out.append((impath, label, classname, caption))  # Include caption in the split data
        return out

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {"train": train, "val": val, "test": test}

    write_json(split, filepath)
    print(f"Saved split to {filepath}")

def subsample_classes(*args, subsample="all"):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
    """
    assert subsample in ["all", "base", "new"]

    if subsample == "all":
        return args

    dataset = args[0]
    labels = set()
    for item in dataset:
        labels.add(item.label)
    labels = list(labels)
    labels.sort()
    n = len(labels)
    # Divide classes into two halves
    m = math.ceil(n / 2)

    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
    if subsample == "base":
        selected = labels[:m]  # take the first half
    else:
        selected = labels[m:]  # take the second half

    relabeler = {y: y_new for y_new, y in enumerate(selected)}

    output = []
    for dataset in args:
        dataset_new = []
        for item in dataset:
            if item.label not in selected:
                continue
            item_new = Datum(
                impath=item.impath,
                label=relabeler[item.label],
                classname=item.classname,
                caption=item.caption  # Ensure caption is passed along
            )
            dataset_new.append(item_new)
        output.append(dataset_new)

    return output



@DATASET_REGISTRY.register()
class PatternNet(DatasetBase):

    dataset_dir = "PatternNet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        print(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.caption_dir = os.path.join(self.dataset_dir, "Captions")  # Add captions folder
        print(self.image_dir)
        self.split_path = os.path.join(self.dataset_dir, "patternnet.json")
        self.shots_dir = os.path.join(self.dataset_dir, "shots")
        mkdir_if_missing(self.shots_dir)

        if os.path.exists(self.split_path):
            train, val, test = read_split(self.split_path, self.image_dir, self.caption_dir)
        else:
            train, val, test = read_and_split_data(self.image_dir, self.caption_dir, ignored=None)
            save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.shots_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
