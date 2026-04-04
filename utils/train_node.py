import os
import sys
from omegaconf import OmegaConf
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pathlib import Path
from models.hypernetwork import Hypernet, NeuralODE, TESTNODE
from models.networks import TaskEmbeddingModel
from algo_testnode import testNODE
from algo import hyperNODE
from dataset import ElasticNODEdataset
from pytorch_lightning.loggers import WandbLogger
# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='algo/configs/config_node.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = OmegaConf.load(file)
            OmegaConf.resolve(config)
        except Exception as exc:
            print(f"Error loading YAML file: {exc}")

    Path(f"{config.save_path}").mkdir(exist_ok=True, parents=True)
    wandb_logger = WandbLogger(project="elastic-neural-ode", name="enode_v1", save_dir="/tmp")
    run_id = wandb_logger.experiment.id
    run_name = wandb_logger.experiment.name
    ckpt_dir = os.path.join(config.save_path, f"{run_name}-{run_id}")
    print("======= The ckpt dir: ", ckpt_dir)
    
    seed_everything(config['exp_params']['manual_seed'], True)

    # model = Hypernet(NeuralODE(in_dim=data_size, out_dim=data_size, hidden_dim=128), 
    #                 ftask_dim=2, 
    #                 **config['Hypernet'])
    # model = TESTNODE(in_dim=data_size, out_dim=data_size, hidden_dim=128)
    model = TESTNODE(**config['MLP_params'])

    experiment = testNODE(hyper_model=model, params=config['exp_params'])

    data = ElasticNODEdataset(**config["dataset_params"], pin_memory=True)
    data.setup()
        
    runner = Trainer(logger=wandb_logger,
                     callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=5, 
                                        dirpath = ckpt_dir, 
                                        monitor= "val_loss",
                                        save_last= True),
                    ],
                    **config['trainer_params'])        
      
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__=="__main__":
    main()