from lightning.pytorch.cli import LightningCLI
import copy

from models import LITMODEL
from datasets import LITDATASET
from typing import Any, Dict, List
import os


class MyLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, save_config_callback=None, **kwargs)
    def setup_parser(
        self, add_subcommands: bool, main_kwargs: Dict[str, Any], subparser_kwargs: Dict[str, Any]
    ) -> None:
        """overload init_parser."""
        self.parser = self.init_parser(**main_kwargs)
        if add_subcommands:
            self._subcommand_method_arguments: Dict[str, List[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)
        self.parser.add_argument("--debug", type=bool, default=False) # default
        self.parser.add_argument("--use_earlystop", type=bool, default=False) # early stopping
        self.parser.add_argument("--compute_metrics", type=bool, default=False) # compute metrics after fit
        self.parser.add_argument("--use_wandb", type=bool, default=False) # default logger

    def before_instantiate_classes(self) -> None:
        """overwrite to run some code before instantiating the classes."""
        # set args for test
        if self.config.subcommand == 'test':
            if self.config.debug:
                self.config.test.trainer.devices = 1
                self.config.test.trainer.logger = "null"
                self.config.test.trainer.limit_test_batches = 10
            return

        # set args for fit
        # set callback dicts
        my_lr_monitor_dict = {
            'class_path': 'lightning.pytorch.callbacks.lr_monitor.LearningRateMonitor',
            'init_args': {'logging_interval': 'epoch'}
        }

        my_early_stopping_dict = {
            'class_path': 'lightning.pytorch.callbacks.early_stopping.EarlyStopping',
            'init_args': {'monitor': 'val_loss', 'patience': 10, 'mode': 'min'}
        }
        # if  os.getenv("MLP_TASK_NAME") is not None:
        #     run_name = f"lr{self.config.fit.model.init_args.lr}_bs{self.config.fit.data.init_args.batch_size}"
        #     task_name = os.getenv("MLP_TASK_NAME")
        #     if task_name:  # for volc task
        #         checkpoint_root = os.getenv("CHECKPOINT_ROOT") or "/iag_ad_01/ad/mayicheng"
        #         dirpath = os.path.join(checkpoint_root, f"{task_name}_{run_name}")
        #     my_checkpoint_dict = {
        #         'class_path': 'lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint',
        #         'init_args': {'save_top_k': 5, 'save_last': True, 'monitor': 'train_loss', 'mode': 'min', 'dirpath': dirpath, 'filename': "{epoch:02d}-{train_loss:.4f}", 'save_on_train_epoch_end': True}
        #     }
        # else:
        my_checkpoint_dict = {
            'class_path': 'lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint',
            'init_args': {'save_top_k': 5, 'save_last': True,'monitor': 'train_loss', 'mode': 'min', 'filename': "{epoch:02d}-{train_loss:.4f}", 'save_on_train_epoch_end': True}
        }
        # default args
        self.config.fit.trainer.strategy = "ddp_find_unused_parameters_true"
        self.config.fit.trainer.callbacks = [my_checkpoint_dict ,my_lr_monitor_dict]
        
        # optional args
        if self.config.use_earlystop:
            self.config.fit.trainer.callbacks.append(my_early_stopping_dict)

        # set logger
        if self.config.use_wandb:
            run_name = f"lr{self.config.fit.model.init_args.lr}_bs{self.config.fit.data.init_args.batch_size}"
            task_name = os.getenv("MLP_TASK_NAME")
            if task_name:  # for volc task
                run_name = f"{task_name}_{run_name}"
            wandb_project = os.getenv("WANDB_PROJECT", "lit_diffusion")
            wandb_logger_dict = {
                'class_path': 'lightning.pytorch.loggers.wandb.WandbLogger',
                'init_args': {'project': wandb_project, 'name': run_name,'log_model': False}
            }
            self.config.fit.trainer.logger = wandb_logger_dict
        else:
            self.config.fit.trainer.logger = "null"

        # set debug mode
        if self.config.debug:
            self.config.fit.trainer.max_epochs = 1
            self.config.fit.trainer.limit_train_batches = 10
            self.config.fit.trainer.limit_val_batches = 10
            self.config.fit.trainer.limit_test_batches = 10
            self.config.fit.trainer.devices = 1
            self.config.fit.trainer.logger = "null"
            self.config.fit.data.init_args.batch_size = 1
            self.config.fit.data.init_args.num_workers = 1
            print("Debug mode is active. Trainer configuration updated for debugging.")

        self.load_weights_only = self.config.get("fit", {}).get("model", {}).get("init_args", {}).get("load_weights_only", False)
        if self.load_weights_only:
            # 检查是否有ckpt_path配置，并保存到实例属性中
            ckpt_path = None
            if hasattr(self.config, 'fit') and hasattr(self.config.fit, 'ckpt_path'):
                ckpt_path = self.config.fit.ckpt_path
            elif hasattr(self.config, 'ckpt_path'):
                ckpt_path = self.config.ckpt_path
            
            if ckpt_path:
                # 保存ckpt_path到实例属性，供before_fit使用
                self._saved_ckpt_path = ckpt_path
                # 清除配置中的ckpt_path（可能在trainer下或顶层），避免Lightning自动加载
                if hasattr(self.config.fit.trainer, 'ckpt_path'):
                    self.config.fit.trainer.ckpt_path = None
                if hasattr(self.config.fit, 'ckpt_path'):
                    self.config.fit.ckpt_path = None
                if hasattr(self.config, 'ckpt_path'):
                    self.config.ckpt_path = None
                print(f"Saved ckpt_path ({ckpt_path}) and cleared it from config to prevent automatic checkpoint loading.")
            else:
                self._saved_ckpt_path = None
    def before_fit(self):
        """在训练开始前调用，如果load_weights_only=True，手动加载权重"""
        if self.load_weights_only:
            # 使用在before_instantiate_classes中保存的ckpt_path
            # 如果之前没有保存，尝试从config中获取（可能通过命令行参数传入）
            ckpt_path = None
            if hasattr(self, '_saved_ckpt_path') and self._saved_ckpt_path:
                ckpt_path = self._saved_ckpt_path
            else:
                # 如果之前没有保存，尝试从config中获取（可能通过命令行参数传入）
                if hasattr(self.config, 'fit') and hasattr(self.config.fit, 'ckpt_path'):
                    ckpt_path = self.config.fit.ckpt_path
                elif hasattr(self.config, 'ckpt_path'):
                    ckpt_path = self.config.ckpt_path
            
            if ckpt_path:
                # 使用手动加载方法重新创建模型
                model_class = type(self.model)
                # 获取模型初始化参数
                model_init_args = copy.deepcopy(self.config.fit.model.init_args)
                # 移除load_weights_only，避免循环
                model_init_args.pop('load_weights_only', None)
                print(f"loaded {ckpt_path}")
                # 重新创建模型并加载权重
                self.model = model_class.load_weights_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    config=model_init_args.get('config'),
                    **{k: v for k, v in model_init_args.items() if k != 'config'}
                )
                # 清除trainer的ckpt_path，避免Lightning自动加载
                # 直接设置trainer.ckpt_path比修改config更有效
                if hasattr(self, 'trainer') and self.trainer is not None:
                    self.trainer.ckpt_path = None
                    # trainer.model 是只读属性，需要使用私有属性 _model 来设置
                    self.trainer._model = self.model
                    print("Cleared trainer.ckpt_path and updated trainer model reference.")

    def after_fit(self):
        if self.config.compute_metrics:
            self.datamodule.setup('test')
            self.trainer.test(ckpt_path='best', dataloaders=self.datamodule.test_dataloader())
            print("Metrics computed and saved.")
        else:
            print("Metrics computed skipped.")
            # 只在全局 rank 0 进程保存文件，避免多进程竞态条件
            # MLP_ROLE_INDEX: 节点级别的序号，单机多卡时所有进程都是 "0"
            if os.getenv("MLP_TASK_NAME") is not None and self.trainer.is_global_zero:
                best_model_path = self.trainer.checkpoint_callback.best_model_path
                print(f"Save best model path{best_model_path} to best_model_path.txt")
                if os.path.exists("best_model_path.txt"):
                    os.remove("best_model_path.txt")
                with open("best_model_path.txt", "w") as f:
                    f.write(best_model_path)


if __name__ == "__main__":
    cli = MyLightningCLI()
