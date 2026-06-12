from datasets import LITDATASET
from models import LITMODEL
import os
import yaml
import lightning as L
import torch
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint
from utils import load_config

def main():
    # print(LITDATASET.module_dict.keys())
    # print(LITMODEL.module_dict.keys())

    hparams_path = os.path.join(os.getcwd(), 'config', 'hparams', 'hparams.yaml')
    hparams_config = load_config(hparams_path)
    
    data_config_path = os.path.join(os.getcwd(), 'config', 'dataset', 'LitUnifiedDataset.yaml')
    data_config = load_config(data_config_path)
    # DataModule
    data_module = LITDATASET.build(dict(type = data_config['lit_name'], config = data_config, batch_size = hparams_config['batch_size'], num_workers = hparams_config['dataloader_num_workers']))
    # Model
    model_config_path = os.path.join(os.getcwd(), 'config', 'model', 'LitMultiTFModel.yaml')
    model_config = load_config(model_config_path)
    model = LITMODEL.build(dict(type = model_config['lit_name'], config = model_config['model'], lr = hparams_config['lr']))
    #model = torch.compile(model)
    trainer = L.Trainer(fast_dev_run=True, profiler=None, devices=[3], accelerator='gpu', strategy='ddp', max_epochs=2, gradient_clip_val=1.0, 
                        check_val_every_n_epoch=1, enable_progress_bar=True, enable_model_summary=True,
                        callbacks=[StochasticWeightAveraging(swa_lrs=1e-3),
                                   ModelCheckpoint(save_top_k=5, monitor='val_loss', mode='min', filename="{epoch:02d}-{val_loss:.2f}")])
    trainer.fit(model=model, datamodule=data_module)



if __name__ == '__main__':
    main()