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
    
    This is for federated learning or any scenario where you
    already have separate subsets. Hence no call to build_dataset(cfg).
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
        # 1) Make sure we have all the subset lists
        # They are presumably list-of-Datum or similar (from Dassl).
        self._validate_labels(train_x, "train_x")
        self._validate_labels(val, "val")
        self._validate_labels(test, "test")
        self.train_x_list = train_x
        self.val_list = val
        self.test_list = test
        self.cfg = cfg
        all_classnames = set()
        for item in (self.train_x_list + self.val_list + self.test_list):
            all_classnames.add(item.classname)
        self._classnames = sorted(list(all_classnames))

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

        # 3) Build data loaders from these subsets
        self.train_loader_x = None
        if self.train_x_list:
            self.train_loader_x = build_data_loader(
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

        # If you need them, you can define these
        self._num_classes = None
        self._lab2cname = None
    
    def _validate_labels(self, data_subset, subset_name):
        """Ensure all data items have valid labels"""
        if not data_subset:
            return
        for i, item in enumerate(data_subset):
            if not hasattr(item, 'label') or item.label is None:
                raise ValueError(
                    f"Missing label in {subset_name} at index {i}. "
                    "All federated data items must contain labels."
                )
            if not isinstance(item.label, int):
                raise TypeError(
                    f"Invalid label type in {subset_name} at index {i}. "
                    f"Expected int, got {type(item.label)}"
                )
    @property
    def dataset(self):
        """
        Some existing code might try to do: dm.dataset.train_x
        So you can emulate that here if needed.
        We'll just build a simple object:
        """
        ds = SimpleNamespace(
            train_x=self.train_x_list,
            val=self.val_list,
            test=self.test_list,
            train_u=[],
            num_classes=self._num_classes,
            lab2cname=self._lab2cname,
            classnames=self._classnames
        )
        return ds

    @property
    def num_classes(self):
        return self._num_classes if self._num_classes else 0

    """@property
    def lab2cname(self):
        return self._lab2cname if self._lab2cname else {}"""
    @property
    def lab2cname(self):
        """Build proper label-to-classname mapping"""
        if self._lab2cname is None:
            self._lab2cname = {}
            for item in self.train_x_list + self.val_list + self.test_list:
                if item.label not in self._lab2cname:
                    self._lab2cname[item.label] = item.classname
        return self._lab2cname
