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
    """
    Read image folders, optionally rename categories via `rename_map`,
    then split into train/val/test.
    """

    # 1) Define the rename map
    #    Left = original folder name, Right = cleaned (standardized) name.
    rename_map = {
    # UC Merced -> PatternNet (exact or closest equivalent)
    "tenniscourt": "tennis_court",
    "golfcourse": "golf_course",
    "parkinglot": "parking_lot",
    "storagetanks": "storage_tank",
    "mobilehomepark": "mobile_home_park",
    "baseballdiamond": "baseball_field",
    "denseresidential": "dense_residential",
    "sparseresidential": "sparse_residential"  }


    if ignored is None:
        ignored = []
    print("IMAGE DIR ", image_dir)
    categories = listdir_nohidden(image_dir)  # e.g. ["tenniscourt", "harbor", ...]
    print("CATEGORIES ", categories)

    # 2) Drop ignored
    categories = [c for c in categories if c not in ignored]

    # 3) Apply rename_map
    renamed_categories = []
    for cat in categories:
        if cat in rename_map:
            renamed_categories.append(rename_map[cat])
        else:
            renamed_categories.append(cat)
    # If multiple original folders map to the same name, you effectively merge them into one label.

    # Possibly remove duplicates if merges happened
    # If you do want merges, you must handle them carefully at label assignment
    renamed_categories = list(sorted(set(renamed_categories)))

    print("RENAMED CATEGORIES", renamed_categories)

    p_tst = 1 - p_trn - p_val
    print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

    all_data = []
    
    # 4) We must now iterate over each original folder again,
    #    but we also track the cleaned category name for labeling.
    #    Because we used `set(...)` above, you must figure out
    #    how to assign folder -> label in code carefully.
    #    Example approach: we do a second pass using the rename_map again.
    
    cat2label = {cat_name: i for i, cat_name in enumerate(renamed_categories)}

    # Re-list categories from the disk (no set(...) or sorting again)
    categories_disk = sorted(listdir_nohidden(image_dir))

    for folder_name in categories_disk:
        if folder_name in ignored:
            continue
        # Clean name
        if folder_name in rename_map:
            cleaned_name = rename_map[folder_name]
        else:
            cleaned_name = folder_name

        if cleaned_name not in cat2label:
            # If rename_map merges two folder names, we skip if the cleaned name isn't recognized
            # Or you can handle it differently
            print(f"[Warning] Skipping folder {folder_name} => {cleaned_name} not in final list")
            continue
        
        label = cat2label[cleaned_name]  # integer label
        image_category_dir = os.path.join(image_dir, folder_name)
        caption_category_dir = os.path.join(caption_dir, folder_name)
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

            item = Datum(
                impath=image_path,
                label=label,
                classname=cleaned_name,
                caption=caption
            )
            all_data.append(item)

    # 5) Split into train/val/test
    # random.shuffle(all_data) if you want a random shuffle
    num_total = len(all_data)
    num_trn = int(p_trn * num_total)
    num_val = int(p_val * num_total)

    train = all_data[:num_trn]
    val = all_data[num_trn : num_trn + num_val]
    test = all_data[num_trn + num_val :]

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
            out.append((impath, label, classname, caption))
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
    """
    import math
    from dassl.data.datasets import Datum

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
    m = math.ceil(n / 2)

    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
    if subsample == "base":
        selected = labels[:m]  # take the first half
    else:
        selected = labels[m:]  # second half

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
                caption=item.caption
            )
            dataset_new.append(item_new)
        output.append(dataset_new)

    return output


@DATASET_REGISTRY.register()
class Ucmerced(DatasetBase):

    dataset_dir = "Ucmerced"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        print(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Images")
        self.caption_dir = os.path.join(self.dataset_dir, "Captions")  # Add captions folder
        print(self.image_dir)
        self.split_path = os.path.join(self.dataset_dir, "Ucmerced.json")
        self.shots_dir = os.path.join(self.dataset_dir, "shots")
        mkdir_if_missing(self.shots_dir)

        # 1) Load or create the train/val/test split JSON
        if os.path.exists(self.split_path):
            train, val, test = read_split(self.split_path, self.image_dir, self.caption_dir)
        else:
            train, val, test = read_and_split_data(
                self.image_dir, self.caption_dir, ignored=None
            )
            save_split(train, val, test, self.split_path, self.image_dir)

        # 2) Handle few-shot
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

        # 3) Optionally subsample classes
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = subsample_classes(train, val, test, subsample=subsample)

        # 4) Initialize the DatasetBase parent class
        super().__init__(train_x=train, val=val, test=test)
