# import os
# import pickle

# from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from dassl.utils import mkdir_if_missing

# from .oxford_pets import OxfordPets
# from .dtd import DescribableTextures as DTD

# # NEW_CNAMES = {
# #     'agriculture land':  'commercial land'  'industrial land'  'public service land'  'residential land'  'transportation land'  'unutilized land'  'water area'
# # }

# NEW_CNAMES = {}


# @DATASET_REGISTRY.register()
# class Mlrs(DatasetBase):

#     dataset_dir = "Mlrs"

#     def __init__(self, cfg):
#         root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
#         self.dataset_dir = os.path.join(root, self.dataset_dir)
#         self.image_dir = os.path.join(self.dataset_dir, "images")
#         self.split_path = os.path.join(self.dataset_dir, "split_zhou_Mlrs.json")
#         self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
#         mkdir_if_missing(self.split_fewshot_dir)

#         if os.path.exists(self.split_path):
#             train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
#         else:
#             train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
#             OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

#         num_shots = cfg.DATASET.NUM_SHOTS
#         if num_shots >= 1:
#             seed = cfg.SEED
#             preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
#             if os.path.exists(preprocessed):
#                 print(f"Loading preprocessed few-shot data from {preprocessed}")
#                 with open(preprocessed, "rb") as file:
#                     data = pickle.load(file)
#                     train, val = data["train"], data["val"]
#             else:
#                 train = self.generate_fewshot_dataset(train, num_shots=num_shots)
#                 val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
#                 data = {"train": train, "val": val}
#                 print(f"Saving preprocessed few-shot data to {preprocessed}")
#                 with open(preprocessed, "wb") as file:
#                     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

#         subsample = cfg.DATASET.SUBSAMPLE_CLASSES
#         train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

#         super().__init__(train_x=train, val=val, test=test)

#     # def update_classname(self, dataset_old):
#     #     dataset_new = []
#     #     for item_old in dataset_old:
#     #         cname_old = item_old.classname
#     #         cname_new = NEW_CNAMES[cname_old]
#     #         item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
#     #         dataset_new.append(item_new)
#     #     return dataset_new




import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {}

@DATASET_REGISTRY.register()
class Mlrs(DatasetBase):

    dataset_dir = "Mlrs"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        # Point to the parallel "captions" folder.
        # This folder should mirror the structure under "images"
        self.caption_dir = os.path.join(self.dataset_dir, "captions")
        
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Mlrs.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # -----------------------------
        # Load or create splits
        # -----------------------------
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # -----------------------------
        # (Optional) Few-shot logic
        # -----------------------------
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-seed_{seed}.pkl"
            )
            
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

        # -----------------------------
        # (Optional) Subsample classes
        # -----------------------------
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train, val, test, subsample=subsample
        )

        # -----------------------------
        #  Add captions to each subset
        # -----------------------------
        train = self._add_captions(train)
        val   = self._add_captions(val)
        test  = self._add_captions(test)

        super().__init__(train_x=train, val=val, test=test)

    def _add_captions(self, data_list):
        """Wrap each item as a Datum with a loaded caption from self.caption_dir."""
        new_list = []
        for old_item in data_list:
            cap = self._load_caption(old_item.impath)
            new_list.append(
                Datum(
                    impath=old_item.impath,
                    label=old_item.label,
                    classname=old_item.classname,
                    caption=cap
                )
            )
        return new_list

    def _load_caption(self, image_path):
        """
        Converts the image path (under self.image_dir) 
        to a corresponding .txt path in self.caption_dir,
        then reads the contents as the caption.
        """
        # e.g. images/island/island_1398.jpg
        rel_path = os.path.relpath(image_path, self.image_dir)
        # e.g. captions/island/island_1398.txt
        caption_path = os.path.join(self.caption_dir, rel_path)
        caption_path = os.path.splitext(caption_path)[0] + '.txt'
        
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                return f.read().strip()
        
        # If you prefer a fallback for missing files,
        # you could return "" or something else.
        raise FileNotFoundError(f"Caption file not found for {caption_path}")
