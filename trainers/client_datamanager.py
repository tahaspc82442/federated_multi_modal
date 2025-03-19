# trainers/client_data_manager.py

import numpy as np
import os
import torch
from dassl.data.data_manager import build_transform, build_data_loader
from dassl.utils import mkdir_if_missing
from types import SimpleNamespace

class ClientDataManager:
    """
    A minimal DataManager-like class that directly accepts
    train_x, val, test subsets (already partitioned).

    - Each subset is a list of Datum objects (from Dassl).
    - We build the PyTorch DataLoader for train/val/test.
    - Provides .train_loader, .val_loader, .test_loader
    - Also provides .dataset, which has .train_x, .val, .test, etc.
    """

    def __init__(
        self,
        train_x,
        val,
        test,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # 1) Save the subsets
        self._validate_labels(train_x, "train_x")
        self._validate_labels(val, "val")
        self._validate_labels(test, "test")
        self.train_x_list = train_x
        self.val_list = val
        self.test_list = test

        self.cfg = cfg

        # Gather classnames from all subsets
        all_classnames = set()
        for item in (train_x + val + test):
            all_classnames.add(item.classname)
        self._classnames = sorted(list(all_classnames))

        # By default, number of classes = #unique classnames
        self._num_classes = len(self._classnames)
        # We'll build a label->classname dict on demand
        self._lab2cname = None

        # 2) Build transforms
        if custom_tfm_train is None:
            self.tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            self.tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            self.tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            self.tfm_test = custom_tfm_test

        # 3) Build data loaders
        # Use build_data_loader from Dassl, which can handle many sampler configs
        self.train_loader = None
        if self.train_x_list:
            self.train_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=self.train_x_list,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=self.tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        self.val_loader = None
        if self.val_list:
            self.val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.val_list,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=self.tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        self.test_loader = None
        if self.test_list:
            self.test_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.test_list,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=self.tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

    def _validate_labels(self, data_subset, subset_name):
        """Ensure all data items have valid integer labels."""
        if not data_subset:
            return
        for i, item in enumerate(data_subset):
            if not hasattr(item, 'label'):
                raise ValueError(
                    f"Missing 'label' attribute in {subset_name} at index {i}."
                )
            if not isinstance(item.label, int):
                raise TypeError(
                    f"Invalid label type in {subset_name} at index {i}. "
                    f"Expected int, got {type(item.label)}"
                )

    @property
    def dataset(self):
        """
        Some code might do: self.dm.dataset.train_x
        Return a namespace that emulates the typical Dassl dataset structure.
        """
        ds = SimpleNamespace(
            train_x=self.train_x_list,
            val=self.val_list,
            test=self.test_list,
            train_u=[],  # optional
            num_classes=self.num_classes,
            lab2cname=self.lab2cname,
            classnames=self._classnames
        )
        return ds

    @property
    def num_classes(self):
        """Number of classes derived from unique classnames."""
        return self._num_classes

    @property
    def lab2cname(self):
        """
        If not explicitly provided, build a label->classname map
        from the data lists. This assumes each label is consistent
        across train/val/test.
        """
        if self._lab2cname is None:
            self._lab2cname = {}
            # For each item, store {label -> classname}
            all_data = self.train_x_list + self.val_list + self.test_list
            for item in all_data:
                if item.label not in self._lab2cname:
                    self._lab2cname[item.label] = item.classname
        return self._lab2cname




# class FedCLIPDatum:
#     def __init__(self, impath, label, classname):
#         self.impath = impath
#         self.label = label
#         self.classname = classname
#         self.prompt = f"a satellite image of {classname}"

# class CLIPDataManager:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.preprocess = self._clip_preprocess()
        
#     def _clip_preprocess(self):
#         return clip.load(self.cfg.MODEL.BACKBONE.NAME)[1]
    
#     def get_loader(self, data, batch_size, is_train):
#         def collate_fn(batch):
#             images = []
#             texts = []
#             labels = []
            
#             for d in batch:
#                 # Image processing
#                 img = Image.open(d.impath).convert("RGB")
#                 images.append(self.preprocess(img))
                
#                 # Text processing
#                 texts.append(clip.tokenize(d.prompt))
                
#                 labels.append(d.label)
            
#             return {
#                 'images': torch.stack(images),
#                 'texts': torch.cat(texts),
#                 'labels': torch.tensor(labels)
#             }
        
#         return torch.utils.data.DataLoader(
#             data,
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#             shuffle=is_train
#         )