


import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "AnnualCrop": "Annual_Crop_Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous_Vegetation_Land",
    "Highway": "Road",
    "Industrial": "Industrial_Buildings",
    "Pasture": "Pasture_Land",
    "PermanentCrop": "Permanent_Crop_Land",
    "Residential": "Residential_Buildings",
    "River": "River",
    "SeaLake": "Sea",
}
@DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):
    dataset_dir = "eurosat"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.caption_dir = os.path.join(self.dataset_dir, "captions")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")

        # Original EuroSAT initialization
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # Add existing captions
        # train = self._add_captions(train)
        # val = self._add_captions(val)
        # test = self._add_captions(test)

        # Rest of original EuroSAT code
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
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
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        train = self._add_captions(train)
        val = self._add_captions(val)
        test = self._add_captions(test)

        super().__init__(train_x=train, val=val, test=test)

    def _add_captions(self, data):
        """Load existing captions from predefined directory structure"""
        return [
            Datum(
                impath=datum.impath,
                label=datum.label,
                classname=datum.classname,
                caption=self._load_caption(datum.impath)
            ) for datum in data
        ]

    def _load_caption(self, image_path):
        """Directly load caption from parallel directory structure"""
        # Convert image path to caption path
        rel_path = os.path.relpath(image_path, self.image_dir)
        caption_path = os.path.join(self.caption_dir, rel_path)
        caption_path = os.path.splitext(caption_path)[0] + '.txt'
        
        # Load caption with existence check
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                return f.read().strip()
        raise FileNotFoundError(f"Caption file missing: {caption_path}")

    def update_classname(self, dataset_old):
        """Preserve captions while updating classnames"""
        return [
            Datum(
                impath=item.impath,
                label=item.label,
                classname=NEW_CNAMES[item.classname],
                caption=item.caption
            )
            for item in dataset_old
        ]