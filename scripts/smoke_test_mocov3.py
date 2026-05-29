"""
CPU 스모크 테스트 — MoCo v3 + ViT-S/16 통합 검증.

확인 항목:
1. MoCoV3 빌드 (trainable params, total params, feature_dim)
2. patch_embed.proj 고정 여부 (requires_grad=False)
3. AdamW optimizer param groups 분리 (decay vs no_decay)
4. forward → 유한 loss, feature_std (collapse 아님), tau (cosine)
5. backward 정상
6. _update_target() 후 momentum encoder 갱신 확인
7. checkpoint 저장 + fresh backbone 재로드 (missing=0, unexpected=0)
8. 회귀: MoCo v2 빌드 + forward + backward 정상

실행:
    python scripts/smoke_test_mocov3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

import torch
import yaml

# ---- 1) Config 로드 (테스트용 mini batch) ----
with open('configs/mocov3_vits.yaml') as f:
    cfg = yaml.safe_load(f)

print('=' * 70)
print('[1] MoCoV3 빌드 (CPU)')
print('=' * 70)

from ssl_lib.models.mocov3 import MoCoV3
from ssl_lib.utils.schedulers import build_optimizer, CosineLRScheduler

model = MoCoV3(cfg)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  total      : {total/1e6:.1f}M')
print(f'  trainable  : {trainable/1e6:.1f}M  (frozen momentum encoder + frozen patch_embed.proj 제외)')
print(f'  feat_dim   : {model.backbone.feature_dim}')
assert model.backbone.feature_dim == 384, 'ViT-S embed_dim은 384여야 함'

print()
print('=' * 70)
print('[2] patch_embed.proj frozen 확인')
print('=' * 70)
flags = [p.requires_grad for p in model.backbone.net.patch_embed.proj.parameters()]
print(f'  patch_embed.proj.requires_grad = {flags}')
assert all(not f for f in flags), 'frozen patch embed 실패'
print('  → frozen OK')

print()
print('=' * 70)
print('[3] AdamW param groups')
print('=' * 70)
optimizer = build_optimizer(model, cfg)
print(f'  optimizer type    : {type(optimizer).__name__}')
print(f'  num param_groups  : {len(optimizer.param_groups)}')
for i, g in enumerate(optimizer.param_groups):
    nparams = sum(p.numel() for p in g['params'])
    print(f'    group {i}: wd={g["weight_decay"]}, n_params={nparams:,}')
assert isinstance(optimizer, torch.optim.AdamW), 'AdamW가 아님'
assert len(optimizer.param_groups) == 2, 'decay/no_decay 2 groups여야 함'
wds = sorted({g['weight_decay'] for g in optimizer.param_groups})
assert wds == [0.0, 0.1], f'expected [0.0, 0.1], got {wds}'
# patch_embed.proj가 optimizer에 포함되지 않았는지
opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
for p in model.backbone.net.patch_embed.proj.parameters():
    assert id(p) not in opt_param_ids, 'frozen patch_embed.proj가 optimizer에 들어감!'
print('  → frozen patch_embed.proj는 optimizer에서 제외 OK')

print()
print('=' * 70)
print('[4] Forward + [5] Backward (CPU, B=8, 96x96)')
print('=' * 70)
torch.manual_seed(0)
B = 8
x1 = torch.randn(B, 3, 96, 96)
x2 = torch.randn(B, 3, 96, 96)

# eval mode → BatchNorm running stats 영향 안 받고 안정적
# 하지만 학습 동작을 보려면 train. 작은 배치라도 OK.
model.train()
model.set_ema_tau(current_step=0.0, total_steps=300)
tau_start = model._current_tau
loss, log = model(x1, x2)
print(f'  loss       : {log["loss"]:.4f}  (finite={torch.isfinite(loss).item()})')
print(f'  feat_std   : {log["feature_std"]:.4f}  (collapse면 <0.02)')
print(f'  ema_tau(0) : {log["ema_tau"]:.6f}')
assert torch.isfinite(loss).item(), 'loss가 발산'
assert log['feature_std'] > 0.01, 'collapse 의심 (feat_std 너무 작음)'

loss.backward()
# patch_embed.proj는 grad가 None이어야 함
for p in model.backbone.net.patch_embed.proj.parameters():
    assert p.grad is None, 'frozen patch embed에 grad가 흘렀음!'
print('  backward   : OK, frozen patch_embed.proj에 grad 없음')

print()
print('=' * 70)
print('[6] EMA τ schedule (cosine) + _update_target()')
print('=' * 70)
# 학습 진행 시 tau 변화 확인
for prog in [0.0, 0.25, 0.5, 0.75, 1.0]:
    model.set_ema_tau(current_step=prog * 300, total_steps=300)
    print(f'  progress={prog:.2f} → tau={model._current_tau:.6f}')
# tau는 0.99에서 1.0으로 증가해야 함
model.set_ema_tau(current_step=0.0, total_steps=300)
tau0 = model._current_tau
model.set_ema_tau(current_step=300, total_steps=300)
tau_end = model._current_tau
assert tau0 < tau_end <= 1.0 + 1e-9, f'cosine 증가 실패: {tau0} → {tau_end}'

# _update_target: base와 momentum이 다르게 가야 함 (deepcopy 직후엔 같음)
optimizer.zero_grad(set_to_none=True)
loss, _ = model(x1, x2)
loss.backward()
# AdamW step 하기 전 base/momentum 비교 — deepcopy 직후엔 동일해야 함
base_p = next(model.base_encoder.parameters()).detach().clone()
mom_p = next(model.momentum_encoder.parameters()).detach().clone()
diff_before_step = (base_p - mom_p).abs().max().item()
print(f'  base vs momentum (step 전): max_diff={diff_before_step:.2e}')

optimizer.step()
model.set_ema_tau(current_step=10, total_steps=300)
model._update_target()
base_p2 = next(model.base_encoder.parameters()).detach().clone()
mom_p2 = next(model.momentum_encoder.parameters()).detach().clone()
diff_after = (base_p2 - mom_p2).abs().max().item()
print(f'  base vs momentum (step+EMA 후): max_diff={diff_after:.2e}')
assert diff_after > 0, 'EMA 업데이트 후에도 base==momentum'

print()
print('=' * 70)
print('[7] Checkpoint save/load roundtrip')
print('=' * 70)
from ssl_lib.utils.checkpoint import save_checkpoint
from ssl_lib.models.backbone import build_backbone

with tempfile.TemporaryDirectory() as tmp:
    save_checkpoint(
        model=model,
        optimizer=None,
        epoch=1,
        output_dir=tmp,
        extra={'config': cfg},
    )
    ckpt_files = list(Path(tmp).glob('backbone_ep*.pth'))
    assert ckpt_files, 'backbone_ep*.pth 저장 실패'
    print(f'  saved: {ckpt_files[0].name}')

    # fresh backbone 빌드 + 로드
    fresh_bb = build_backbone(cfg)
    state = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
    missing, unexpected = fresh_bb.load_state_dict(state['backbone_state_dict'], strict=False)
    print(f'  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
    assert len(missing) == 0 and len(unexpected) == 0, f'state_dict mismatch: missing={missing}, unexpected={unexpected}'

    # forward 검증
    fresh_bb.eval()
    with torch.no_grad():
        out = fresh_bb(x1)
    print(f'  fresh backbone forward → {tuple(out.shape)}')
    assert out.shape == (B, 384)

print()
print('=' * 70)
print('[8] 회귀: MoCo v2 빌드 + forward + backward')
print('=' * 70)
with open('configs/mocov2_r50.yaml') as f:
    cfg_v2 = yaml.safe_load(f)
# CPU에서 batch_size 4로 축소
cfg_v2['training']['batch_size'] = 4
cfg_v2['mocov2']['queue_size'] = 8  # 4의 배수 → 작게
cfg_v2['backbone']['gradient_checkpoint'] = False

from ssl_lib.models.mocov2 import MoCoV2
m2 = MoCoV2(cfg_v2)
n2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
print(f'  MoCoV2 trainable: {n2/1e6:.1f}M')
m2.train()
xa = torch.randn(4, 3, 96, 96)
xb = torch.randn(4, 3, 96, 96)
loss2, log2 = m2(xa, xb)
print(f'  loss={log2["loss"]:.4f}, feat_std={log2.get("feature_std", 0.0):.4f}')
assert torch.isfinite(loss2).item()
loss2.backward()
print('  MoCo v2 forward+backward OK')

print()
print('=' * 70)
print('ALL SMOKE TESTS PASSED')
print('=' * 70)
