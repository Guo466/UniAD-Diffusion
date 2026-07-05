import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

warnings.filterwarnings("ignore")

from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import PlanningMetric


def custom_single_gpu_test(model, data_loader, show=False, out_dir=None):
    """单卡版本的 UniAD 推理，支持 planning / occ 指标累积（与 custom_multi_gpu_test 对齐）。"""
    model.eval()

    # 兼容 MMDataParallel（model.module）和裸模型两种情况
    _model = model.module if hasattr(model, 'module') else model

    # 用最可靠的方式判断：直接检查 planning_head 属性是否存在且非 None
    _ph_val = getattr(_model, 'planning_head', None)
    eval_planning = (_ph_val is not None)
    eval_occ = (getattr(_model, 'occ_head', None) is not None)

    # 诊断信息同时写到文件，避免终端滚屏覆盖
    _diag_msg = (
        f'[eval_init] model={type(_model).__name__}  '
        f'eval_planning={eval_planning}  eval_occ={eval_occ}  '
        f'planning_head={type(_ph_val).__name__ if _ph_val else None}\n'
    )
    print(_diag_msg)
    import sys
    sys.stdout.flush()
    try:
        with open('/tmp/uniad_eval_diag.txt', 'w') as _f:
            _f.write(_diag_msg)
    except Exception:
        pass

    if eval_planning:
        planning_metrics = PlanningMetric().cuda()

    if eval_occ:
        from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
            IntersectionOverUnion, PanopticMetric)
        EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {k: IntersectionOverUnion(n_classes).cuda() for k in EVALUATION_RANGES}
        panoptic_metrics = {k: PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
                            for k in EVALUATION_RANGES}
        num_occ = 0

    bbox_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # ---- 规划指标累积 ----
        if eval_planning and 'planning' in result[0]:
            try:
                planning_gt   = result[0]['planning']['planning_gt']
                seg           = planning_gt['segmentation']        # list[Tensor(B,T+1,H,W)]
                sdc_plan      = planning_gt['sdc_planning']        # list[Tensor(B,n_modes,T,3)] or Tensor
                sdc_mask      = planning_gt['sdc_planning_mask']   # 同上
                command       = planning_gt['command']
                pred_traj     = result[0]['planning']['result_planning']['sdc_traj']  # (B,T,2)

                # sdc_plan/sdc_mask 可能是 list（数据并行）或直接 Tensor
                sp = sdc_plan[0] if isinstance(sdc_plan, (list, tuple)) else sdc_plan
                sm = sdc_mask[0] if isinstance(sdc_mask, (list, tuple)) else sdc_mask
                sg = seg[0]      if isinstance(seg,      (list, tuple)) else seg

                # sp: (B, n_modes, T, 3) 或 (n_modes, T, 3)
                # 取 command 对应的模态（与 multi_gpu_test 保持一致取第0个 sample 的第0个模态）
                if sp.dim() == 4:          # (B, n_modes, T, 3)
                    sp_traj = sp[0, 0:1, :6, :2]   # (1, 6, 2)
                    sm_traj = sm[0, 0:1, :6, :2]   # (1, 6, 2)
                elif sp.dim() == 3:        # (n_modes, T, 3)
                    sp_traj = sp[0:1, :6, :2]       # (1, 6, 2)
                    sm_traj = sm[0:1, :6, :2]       # (1, 6, 2)
                else:
                    sp_traj = sp[:, :6, :2]
                    sm_traj = sm[:, :6, :2]

                # seg: (T, H, W) 或 (B, T, H, W) → 取前 6 个未来帧
                if sg.dim() == 4:
                    seg_fut = sg[0, [1,2,3,4,5,6]]   # (6, H, W)
                else:
                    seg_fut = sg[[1,2,3,4,5,6]]       # (6, H, W)
                seg_fut = seg_fut.unsqueeze(0)         # (1, 6, H, W)

                # pred_traj: (B,T,2) → 取前6步
                pred6 = pred_traj[:, :6, :2]           # (1, 6, 2)

                if i == 0:
                    print(f'\n[PlanMetric debug] pred={pred6.shape}  '
                          f'gt={sp_traj.shape}  mask={sm_traj.shape}  seg={seg_fut.shape}')

                planning_metrics(pred6, sp_traj, sm_traj, seg_fut)

                result[0]['planning_traj']    = pred_traj
                result[0]['planning_traj_gt'] = sdc_plan
                result[0]['command']          = command
            except Exception as e:
                if i == 0:
                    print(f'\n[PlanMetric WARNING] 第{i}帧规划指标累积失败: {e}')

        # ---- OCC 指标累积 ----
        if eval_occ:
            occ_has_invalid = data.get('gt_occ_has_invalid_frame', [None])[0]
            occ_to_eval = (occ_has_invalid is not None and not occ_has_invalid.item()) \
                          and 'occ' in result[0]
            if occ_to_eval:
                num_occ += 1
                for key, grid in EVALUATION_RANGES.items():
                    lim = slice(grid[0], grid[1])
                    iou_metrics[key](
                        result[0]['occ']['seg_out'][..., lim, lim].contiguous(),
                        result[0]['occ']['seg_gt'][..., lim, lim].contiguous())
                    panoptic_metrics[key](
                        result[0]['occ']['ins_seg_out'][..., lim, lim].contiguous().detach(),
                        result[0]['occ']['ins_seg_gt'][..., lim, lim].contiguous())

        # 清理不需要序列化的大字段
        result[0].pop('occ', None)
        result[0].pop('planning', None)

        bbox_results.extend(result)
        prog_bar.update()

    ret = dict(bbox_results=bbox_results)

    if eval_planning:
        ret['planning_results_computed'] = planning_metrics.compute()
        planning_metrics.reset()

    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for pk, val in panoptic_scores.items():
                occ_results[pk] = occ_results.get(pk, []) + [100 * val[1].item()]
            panoptic_metrics[key].reset()
            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()
        occ_results['num_occ']   = num_occ
        occ_results['ratio_occ'] = num_occ / len(dataset)
        ret['occ_results_computed'] = occ_results

    return ret


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = custom_single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            #assert False
            mmcv.dump(outputs, args.out)
            #outputs = mmcv.load(args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    # NOTE: To fix the serialization issue in nuScenes-dev-kit, we adopt this method to skip the pickle steps
    torch.multiprocessing.set_start_method('fork')
    main()