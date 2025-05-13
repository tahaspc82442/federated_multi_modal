import os
import torch
import numpy as np

from dassl.engine import TRAINER_REGISTRY
from dassl.data import DataManager
from dassl.utils import mkdir_if_missing, load_checkpoint

from .maple_fed import MaPLeFederated
from .client_datamanager import ClientDataManager


@TRAINER_REGISTRY.register()
class MaPLeFederatedTester(MaPLeFederated):
    """Modified version of MaPLeFederated that only loads the target dataset for testing.
    This is a simplified version that overrides build_data_loader to avoid loading hard-coded
    datasets (PatternNet, UCMerced, etc.) which are required in the original trainer.
    """
    
    def build_data_loader(self):
        """Override the original build_data_loader method to only load the target dataset"""
        print("Using simplified data loader for single-dataset testing")
        
        # Create a data manager for the target dataset (the one specified in cfg)
        cfg = self.cfg.clone()
        data_manager = DataManager(cfg)
        
        # Store the classnames
        self.lab2cname = {i: name for i, name in enumerate(data_manager.dataset.classnames)}
        
        # Set up a client data manager for testing
        test_client_dm = ClientDataManager(
            train_x=data_manager.dataset.train_x,
            val=data_manager.dataset.val,
            test=data_manager.dataset.test,
            cfg=self.cfg
        )
        
        # Store everything we need
        self.client_data_managers = [test_client_dm]
        
        # Set these to None as they're not used by the aggregator
        self.train_loader_x = None
        self.val_loader = None
        self.test_loader = None
        self.dm = data_manager  # Keep reference to the original data manager
        
        print(f"Loaded dataset: {cfg.DATASET.NAME} with {len(self.lab2cname)} classes")
        print(f"Test set size: {len(data_manager.dataset.test)} images")
        
        # Don't call debug_clients_data to avoid errors with missing datasets
    
    def load_model(self, directory, epoch=None):
        """Modified load_model to handle different checkpoint formats and token size mismatches"""
        print(f"Loading model from {directory} (epoch={epoch})")
        
        if not directory:
            print("Skipping load_model, no pretrained path given")
            return
        
        # Try loading directly
        try:
            if epoch is not None:
                model_file = f"model.pth.tar-{epoch}"
            else:
                model_file = "model-best.pth.tar"
            
            path = os.path.join(directory, model_file)
            
            if os.path.exists(path):
                print(f"Found checkpoint at {path}")
                ckpt = load_checkpoint(path)
                state_dict = ckpt["state_dict"]
                loaded_epoch = ckpt.get("epoch", None)
                
                # Filter out token-related parameters that might cause size mismatches
                # when testing on a dataset with a different number of classes
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    # Skip token prefix and suffix which are class-specific
                    if 'token_prefix' in key or 'token_suffix' in key:
                        print(f"Skipping {key} due to potential class count mismatch")
                        continue
                    filtered_state_dict[key] = value
                
                self.global_weights = filtered_state_dict
                print(f"Loaded weights from '{path}' (epoch={loaded_epoch})")
                
                # Now broadcast the weights to clients with strict=False
                print("Broadcasting weights with strict=False to handle parameter mismatches")
                for client_trainer in self.clients:
                    try:
                        # Load with strict=False to ignore mismatched parameters
                        client_trainer.model.load_state_dict(filtered_state_dict, strict=False)
                        print(f"Successfully loaded filtered weights to client {client_trainer.client_id}")
                    except Exception as e:
                        print(f"Error loading weights to client {client_trainer.client_id}: {e}")
                        return False
                
                return True
            else:
                print(f"Checkpoint not found at {path}")
                
                # Try aggregator subfolder
                subfolder = "MultiModalPromptLearner_Aggregator"
                aggregator_path = os.path.join(directory, subfolder, model_file)
                if os.path.exists(aggregator_path):
                    print(f"Found checkpoint in aggregator subfolder: {aggregator_path}")
                    ckpt = load_checkpoint(aggregator_path)
                    state_dict = ckpt["state_dict"]
                    loaded_epoch = ckpt.get("epoch", None)
                    
                    # Filter out token-related parameters
                    filtered_state_dict = {}
                    for key, value in state_dict.items():
                        if 'token_prefix' in key or 'token_suffix' in key:
                            print(f"Skipping {key} due to potential class count mismatch")
                            continue
                        filtered_state_dict[key] = value
                    
                    self.global_weights = filtered_state_dict
                    print(f"Loaded weights from '{aggregator_path}' (epoch={loaded_epoch})")
                    
                    # Now broadcast the weights to clients with strict=False
                    print("Broadcasting weights with strict=False to handle parameter mismatches")
                    for client_trainer in self.clients:
                        try:
                            # Load with strict=False to ignore mismatched parameters
                            client_trainer.model.load_state_dict(filtered_state_dict, strict=False)
                            print(f"Successfully loaded filtered weights to client {client_trainer.client_id}")
                        except Exception as e:
                            print(f"Error loading weights to client {client_trainer.client_id}: {e}")
                            return False
                    
                    return True
                else:
                    print(f"Checkpoint not found in aggregator subfolder either: {aggregator_path}")
        except Exception as e:
            print(f"Error loading or processing checkpoint: {e}")
        
        print("No valid checkpoint found for testing.")
        return False
        
    def test(self):
        """Override the test method to provide more detailed results"""
        print("\n" + "="*80)
        print("STARTING MODEL EVALUATION")
        print("="*80)
        
        # Ensure we're in eval mode
        for client in self.clients:
            client.model.eval()
            
        # Client 0 is the one we test on
        client = self.clients[0]
        data_manager = self.client_data_managers[0]
        
        # Print dataset info
        dataset_name = self.cfg.DATASET.NAME
        num_classes = len(self.lab2cname)
        test_size = len(data_manager.dataset.test)
        
        print(f"Dataset:       {dataset_name}")
        print(f"Num classes:   {num_classes}")
        print(f"Test set size: {test_size} images")
        
        # Run standard evaluation via client's test method
        with torch.no_grad():
            test_results = client.test()
        
        # Get detailed metrics from the results
        if test_results is not None:
            accuracy = test_results.get('accuracy', 'Not available')
            
            print("\n" + "-"*80)
            print("TEST RESULTS:")
            print("-"*80)
            print(f"Accuracy:      {accuracy}%")
            
            # Display top-5 accuracy if available
            if 'accuracy_top-5' in test_results:
                top5_acc = test_results.get('accuracy_top-5', 'Not available')
                print(f"Top-5 Accuracy: {top5_acc}%")
                
            # Display per-class results if available
            if hasattr(client, 'evaluator') and hasattr(client.evaluator, 'y_true') and hasattr(client.evaluator, 'y_pred'):
                from sklearn.metrics import classification_report
                y_true = client.evaluator.y_true
                y_pred = client.evaluator.y_pred
                
                # Try to get per-class accuracy
                print("\nPer-class accuracy:")
                classnames = [self.lab2cname[i] for i in range(num_classes)]
                try:
                    report = classification_report(
                        y_true, y_pred, 
                        target_names=classnames,
                        output_dict=True
                    )
                    print("\nTop and bottom 5 classes by accuracy:")
                    
                    # Extract class accuracies and sort them
                    class_accuracies = []
                    for i in range(num_classes):
                        class_name = self.lab2cname[i]
                        if class_name in report:
                            precision = report[class_name]['precision'] * 100
                            support = report[class_name]['support']
                            class_accuracies.append((class_name, precision, support))
                    
                    # Sort by accuracy (descending)
                    class_accuracies.sort(key=lambda x: x[1], reverse=True)
                    
                    # Show top 5 classes
                    print("Top 5 classes:")
                    for name, acc, support in class_accuracies[:5]:
                        print(f"  {name}: {acc:.2f}% (samples: {support})")
                    
                    # Show bottom 5 classes
                    print("\nBottom 5 classes:")
                    for name, acc, support in class_accuracies[-5:]:
                        print(f"  {name}: {acc:.2f}% (samples: {support})")
                        
                except Exception as e:
                    print(f"Error generating detailed per-class metrics: {e}")
        else:
            print("No test results returned from evaluation.")
        
        print("="*80)
        print("EVALUATION COMPLETE")
        print("="*80 + "\n")
        
        return test_results 