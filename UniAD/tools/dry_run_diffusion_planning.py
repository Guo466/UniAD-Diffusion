"""
dry_run_diffusion_planning.py
==============================
不依赖 mmdet/mmcv，用随机张量直接验证 DiffusionPlanningHead 核心模块。

测试内容：
  1. 桥接层（BEVToContextBridge、EgoContextBridge）
  2. 训练前向（flow matching loss 计算）
  3. 反向传播（梯度计算）
  4. 推理前向（ODE 采样）
  5. 无 agent 场景鲁棒性

运行方式：
  cd /home/guojinliang/VLA/UniAD
  /home/guojinliang/miniforge3/envs/uniad/bin/python tools/dry_run_diffusion_planning.py
"""

import sys
import os
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def import_diffusion_head():
    """动态 import DiffusionPlanningHead，绕过 mmdet registry 依赖。"""
    import importlib.util

    head_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "projects", "mmdet3d_plugin", "uniad", "dense_heads", "diffusion_planning_head.py"
    )
    spec = importlib.util.spec_from_file_location("diffusion_planning_head", head_path)
    mod  = importlib.util.module_from_spec(spec)

    # ---- mock: mmdet.models.builder ----
    def mock_build_loss(cfg):
        class DummyLoss(nn.Module):
            def forward(self, *args, **kwargs):
                for a in args:
                    if isinstance(a, torch.Tensor):
                        return torch.zeros(1, device=a.device, requires_grad=True)
                return torch.zeros(1, requires_grad=True)
        return DummyLoss()

    class MockRegistry:
        """支持 @HEADS.register_module() 装饰器语法（有/无括号）。"""
        def register_module(self, name=None, force=False, module=None):
            def decorator(cls):
                return cls
            if callable(name):          # @HEADS.register_module 无括号调用
                return name
            if module is not None:
                return module
            return decorator            # @HEADS.register_module() 有括号调用

    mock_builder = types.ModuleType("mmdet.models.builder")
    mock_builder.HEADS      = MockRegistry()
    mock_builder.build_loss = mock_build_loss

    mock_mmdet = types.ModuleType("mmdet")
    mock_mmdet_models = types.ModuleType("mmdet.models")

    sys.modules.setdefault('mmdet',               mock_mmdet)
    sys.modules.setdefault('mmdet.models',         mock_mmdet_models)
    sys.modules.setdefault('mmdet.models.builder', mock_builder)

    # ---- mock: einops（若未安装）----
    try:
        import einops  # noqa
    except ImportError:
        mock_einops = types.ModuleType("einops")

        def rearrange(x, pattern, **kwargs):
            if pattern == '(h w) b d -> b d h w':
                h = kwargs['h']; w = kwargs.get('w', h)
                seq, b, d = x.shape
                return x.reshape(h, w, b, d).permute(2, 3, 0, 1)
            elif pattern == 'b d h w -> (h w) b d':
                b, d, h, w = x.shape
                return x.permute(2, 3, 0, 1).reshape(h * w, b, d)
            elif pattern == 'b p c -> p b c':
                return x.permute(1, 0, 2)
            elif pattern == 'b c h w -> (h w) b c':
                b, c, h, w = x.shape
                return x.permute(2, 3, 0, 1).reshape(h * w, b, c)
            else:
                raise NotImplementedError(f"rearrange pattern not supported: {pattern}")

        mock_einops.rearrange = rearrange
        sys.modules['einops'] = mock_einops

    spec.loader.exec_module(mod)
    return mod


def make_fake_inputs(B, N_agents, embed_dims, planning_steps, device):
    """构造模拟的 UniAD MotionHead 输出和规划 GT。"""
    bev_h = bev_w = 200
    n_layers, n_modes = 3, 6

    bev_embed = torch.randn(bev_h * bev_w, B, embed_dims, device=device)
    outs_motion = {
        'sdc_traj_query':  torch.randn(n_layers, B, n_modes, embed_dims, device=device),
        'sdc_track_query': torch.randn(B, embed_dims, device=device),
        'track_query':     torch.randn(B, N_agents, embed_dims, device=device),
        'bev_pos':         torch.randn(B, embed_dims, bev_h, bev_w, device=device),
    }

    # GT 轨迹：模拟直行（x 方向匀速，每步 0.8m）
    gt_traj = torch.zeros(B, 1, planning_steps, 3, device=device)
    for t in range(planning_steps):
        gt_traj[:, :, t, 0] = (t + 1) * 0.8
    gt_mask = torch.ones(B, 1, planning_steps, 1, device=device)
    command = torch.ones(B, dtype=torch.long, device=device)  # 1 = 直行

    return bev_embed, outs_motion, gt_traj, gt_mask, command


def main():
    print("=" * 65)
    print("  UniAD-Diffusion  DiffusionPlanningHead  端到端验证脚本")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[环境]  torch={torch.__version__}  device={device}")

    # ------------------------------------------------------------------
    # 1. 动态导入模块
    # ------------------------------------------------------------------
    print("\n[1] 导入 DiffusionPlanningHead 模块 ...")
    try:
        mod = import_diffusion_head()
        DiffusionPlanningHead = mod.DiffusionPlanningHead
        BEVToContextBridge    = mod.BEVToContextBridge
        EgoContextBridge      = mod.EgoContextBridge
        DiTBlock              = mod.DiTBlock
        print("    ✓ 导入成功")
    except Exception as e:
        print(f"    ✗ 导入失败: {e}")
        import traceback; traceback.print_exc()
        return False

    B              = 1
    N_agents       = 20
    embed_dims     = 256
    planning_steps = 6
    bev_h = bev_w  = 200

    # ------------------------------------------------------------------
    # 2. 桥接层单元测试
    # ------------------------------------------------------------------
    print("\n[2] 桥接层单元测试 ...")

    try:
        bridge = BEVToContextBridge(
            bev_in_dim=embed_dims, track_in_dim=embed_dims, out_dim=embed_dims,
            n_bev_tokens=64, bev_h=bev_h, bev_w=bev_w, max_agents=N_agents
        ).to(device)
        bev_t   = torch.randn(bev_h * bev_w, B, embed_dims, device=device)
        track_t = torch.randn(B, N_agents, embed_dims, device=device)
        ctx, mask = bridge(bev_t, track_t)
        print(f"    ✓ BEVToContextBridge  context={ctx.shape}  mask={mask.shape}")
        assert ctx.shape[0] == B and ctx.shape[2] == embed_dims
    except Exception as e:
        print(f"    ✗ BEVToContextBridge 失败: {e}")
        import traceback; traceback.print_exc()
        return False

    try:
        ego_br = EgoContextBridge(in_dim=embed_dims, out_dim=embed_dims, n_commands=3).to(device)
        sdc_tq = torch.randn(3, B, 6, embed_dims, device=device)
        sdc_tk = torch.randn(B, embed_dims, device=device)
        cmd    = torch.ones(B, dtype=torch.long, device=device)
        eg_ctx, eg_rout = ego_br(sdc_tq, sdc_tk, cmd)
        print(f"    ✓ EgoContextBridge    ego_ctx={eg_ctx.shape}  ego_routing={eg_rout.shape}")
        assert eg_ctx.shape == (B, 1, embed_dims) and eg_rout.shape == (B, 1, embed_dims)
    except Exception as e:
        print(f"    ✗ EgoContextBridge 失败: {e}")
        import traceback; traceback.print_exc()
        return False

    try:
        block = DiTBlock(dim=embed_dims, heads=8, mlp_ratio=2.0).to(device)
        x_t   = torch.randn(B, planning_steps, embed_dims, device=device)
        ctx_t = torch.randn(B, N_agents + 64, embed_dims, device=device)
        y_t   = torch.randn(B, embed_dims, device=device)
        out   = block(x_t, ctx_t, y_t)
        print(f"    ✓ DiTBlock            in={x_t.shape} → out={out.shape}")
        assert out.shape == x_t.shape
    except Exception as e:
        print(f"    ✗ DiTBlock 失败: {e}")
        import traceback; traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # 3. 构建完整 DiffusionPlanningHead
    # ------------------------------------------------------------------
    print("\n[3] 构建 DiffusionPlanningHead ...")
    try:
        head = DiffusionPlanningHead(
            embed_dims=embed_dims,
            planning_steps=planning_steps,
            output_dim=2,
            bev_h=bev_h, bev_w=bev_w,
            n_bev_tokens=64,
            max_agents=N_agents,
            dit_depth=4,
            dit_heads=8,
            sample_steps=5,
            flow_matching_loss_weight=1.0,
            loss_collision=[dict(type='CollisionLoss', delta=0.0, weight=2.5)],
            loss_planning=None,
            loss_kinematic=None,
            planning_eval=True,
            use_col_optim=False,
            with_adapter=True,
            n_commands=3,
        ).to(device)

        total_p = sum(p.numel() for p in head.parameters())
        train_p = sum(p.numel() for p in head.parameters() if p.requires_grad)
        print(f"    ✓ 构建成功")
        print(f"      总参数量     : {total_p:>12,}  ({total_p/1e6:.2f}M)")
        print(f"      可训练参数量 : {train_p:>12,}  ({train_p/1e6:.2f}M)")

        print("      各子模块参数量：")
        sub_params = sorted(
            [(name, sum(p.numel() for p in m.parameters()))
             for name, m in head.named_children()],
            key=lambda x: -x[1]
        )
        for name, count in sub_params:
            print(f"        {name:<30s}: {count:>10,}  ({count/1e6:.3f}M)")
    except Exception as e:
        print(f"    ✗ 构建失败: {e}")
        import traceback; traceback.print_exc()
        return False

    # ---- 构造输入 ----
    bev_embed, outs_motion, gt_traj, gt_mask, command = make_fake_inputs(
        B, N_agents, embed_dims, planning_steps, device
    )
    print(f"\n[4] 输入张量 shapes：")
    print(f"      bev_embed      : {bev_embed.shape}  (H*W={bev_h*bev_w}, B={B}, D={embed_dims})")
    print(f"      sdc_traj_query : {outs_motion['sdc_traj_query'].shape}")
    print(f"      sdc_track_query: {outs_motion['sdc_track_query'].shape}")
    print(f"      track_query    : {outs_motion['track_query'].shape}")
    print(f"      gt_traj        : {gt_traj.shape}")
    print(f"      command        : {command.tolist()}")

    # ------------------------------------------------------------------
    # 4. 训练前向（Flow Matching）
    # ------------------------------------------------------------------
    print("\n[5] 训练前向（forward_train）...")
    head.train()
    t0 = time.time()
    try:
        ret = head.forward_train(
            bev_embed=bev_embed,
            outs_motion=outs_motion,
            sdc_planning=gt_traj,
            sdc_planning_mask=gt_mask,
            command=command,
            gt_future_boxes=None,
        )
        elapsed = (time.time() - t0) * 1000
        losses  = ret['losses']
        print(f"    ✓ 前向成功  耗时 {elapsed:.1f}ms")
        print(f"      损失字典：")
        for k, v in losses.items():
            print(f"        {k} = {v.item():.6f}")
        assert 'sdc_traj'     in ret['outs_motion'], "缺少 sdc_traj"
        assert 'sdc_traj_all' in ret['outs_motion'], "缺少 sdc_traj_all"
        print(f"      sdc_traj shape : {ret['outs_motion']['sdc_traj'].shape}  ✓ 接口兼容")
    except Exception as e:
        print(f"    ✗ 训练前向失败: {e}")
        import traceback; traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # 5. 反向传播
    # ------------------------------------------------------------------
    print("\n[6] 反向传播（backward）...")
    try:
        total_loss = sum(losses.values())
        total_loss.backward()
        print(f"    ✓ 反向传播成功   total_loss = {total_loss.item():.6f}")
        grad_ok  = sum(1 for _, p in head.named_parameters()
                       if p.grad is not None and not torch.isnan(p.grad).any())
        grad_nan = sum(1 for _, p in head.named_parameters()
                       if p.grad is not None and torch.isnan(p.grad).any())
        print(f"      有效梯度参数：{grad_ok}  NaN梯度参数：{grad_nan}")
        if grad_nan == 0:
            print("      ✓ 无 NaN 梯度")
        else:
            print("      ⚠ 存在 NaN 梯度，请检查数值稳定性")
    except Exception as e:
        print(f"    ✗ 反向传播失败: {e}")
        import traceback; traceback.print_exc()
        return False

    # ------------------------------------------------------------------
    # 6. 推理前向（ODE 采样）
    # ------------------------------------------------------------------
    print("\n[7] 推理前向（forward_test）...")
    head.eval()
    with torch.no_grad():
        t0 = time.time()
        try:
            result = head.forward_test(
                bev_embed=bev_embed,
                outs_motion=outs_motion,
                outs_occflow={},
                command=command,
            )
            elapsed  = (time.time() - t0) * 1000
            pred_traj = result['sdc_traj']   # (1, B, planning_steps, 2) 或 (1, planning_steps, 2)
            print(f"    ✓ 推理成功  耗时 {elapsed:.1f}ms")
            print(f"      sdc_traj shape : {pred_traj.shape}")
            print(f"      预测轨迹（ego 坐标系，单位：米）：")
            # 支持 (1,1,T,2)、(1,T,2) 等多种 shape
            traj_flat = pred_traj.reshape(-1, pred_traj.shape[-1]).cpu().numpy()  # (T, 2)
            for i in range(min(len(traj_flat), planning_steps)):
                xy = traj_flat[i]
                print(f"        t={(i+1)*0.5:.1f}s : x={xy[0]:+.3f}m  y={xy[1]:+.3f}m")
        except Exception as e:
            print(f"    ✗ 推理失败: {e}")
            import traceback; traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # 7. 无 agent 鲁棒性测试
    # ------------------------------------------------------------------
    print("\n[8] 无 agent 场景测试（track_query=None）...")
    head.eval()
    outs_no_agent = {k: v for k, v in outs_motion.items() if k != 'track_query'}
    with torch.no_grad():
        try:
            result_na = head.forward_test(
                bev_embed=bev_embed,
                outs_motion=outs_no_agent,
                outs_occflow={},
                command=command,
            )
            print(f"    ✓ 无 agent 推理成功  shape={result_na['sdc_traj'].shape}")
        except Exception as e:
            print(f"    ✗ 无 agent 推理失败: {e}")
            import traceback; traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # 最终摘要
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ✅ 所有测试通过！DiffusionPlanningHead 端到端验证完成。")
    print("=" * 65)
    print("""
  融合架构概览（方案A）：

  UniAD 已有模块（不变）          → 新增 DiffusionPlanningHead
  ─────────────────────────          ─────────────────────────────────
  BEVFormer                            BEVToContextBridge
   bev_embed(H*W,B,256) ────────────→  BEV 压缩：200×200 → 8×8=64 token
                                        Agent Encoder（2层 Transformer）
  TrackHead                              context: (B, N_agent+64, 256)
   track_query(B,N,256) ────────────→ 
                                       EgoContextBridge
  MotionHead                             ego_context: (B, 1, 256)
   sdc_traj_query  ─────────────────→   ego_routing:  (B, 1, 256)
   sdc_track_query ─────────────────→   (融合 command 导航嵌入)
  
  OccHead（可选）                       DiT 解码器（4层 DiTBlock）
                                         训练：Flow Matching MSE Loss
                                         推理：Euler ODE（5步采样）
                                          ↓
                                         ego 未来轨迹：(B, 6, 2)
                                         6步 × 0.5s = 3秒规划视野
  """)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)