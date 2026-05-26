"""
STL10 unlabeled 데이터셋 wrapper.

torchvision.datasets.STL10을 사용. split='unlabeled'로 100k 이미지 로드.
two-view augmentation을 적용해서 (view1, view2) 튜플 반환.

참고: torchvision STL10은 처음 호출 시 자동 다운로드 (약 2.6GB).
"""
from typing import Tuple, Callable, Optional

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10


class STL10UnlabeledDataset(Dataset):
    """
    STL10 unlabeled split (100k images, 96x96 RGB).
    
    SSL pretraining 전용. Two-view augmentation을 적용해
    같은 이미지의 서로 다른 두 view를 반환.
    """
    
    def __init__(
        self,
        root: str,
        transform: Callable,
        download: bool = True,
    ):
        """
        Args:
            root: 데이터 저장 경로 (예: "./data")
            transform: TwoViewTransform 또는 MultiCropTransform 인스턴스.
                       전자는 (v1, v2) 튜플, 후자는 List[Tensor]를 반환.
            download: 데이터가 없으면 자동 다운로드.
        """
        self.dataset = STL10(
            root=root,
            split="unlabeled",
            transform=None,  # transform은 __getitem__에서 직접 적용
            download=download,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # STL10 unlabeled은 (img, -1) 반환 (라벨 없음)
        img, _ = self.dataset[idx]
        # transform output은 TwoView면 tuple, MultiCrop이면 list.
        # DataLoader의 default_collate는 둘 다 지원 → 그대로 반환.
        return self.transform(img)


def build_stl10_loader(
    cfg: dict,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Config dict로부터 DataLoader 생성.
    
    필수 cfg keys:
        cfg["data"]["root"]          : 데이터 루트 경로
        cfg["data"]["num_workers"]   : worker 수
        cfg["training"]["batch_size"]: batch size
    
    Args:
        cfg: YAML에서 로드한 config dict.
        transform: 이미 빌드된 transform. None이면 build_train_transform 호출.
    
    Returns:
        DataLoader (shuffle=True, drop_last=True, pin_memory=True).
    """
    if transform is None:
        aug_type = cfg.get("augmentation", {}).get("type", "two_view")
        if aug_type == "multicrop":
            from .multicrop_transforms import build_multicrop_transform
            transform = build_multicrop_transform(cfg)
        else:
            from .transforms import build_train_transform
            transform = build_train_transform(cfg)
    
    dataset = STL10UnlabeledDataset(
        root=cfg["data"]["root"],
        transform=transform,
        download=cfg["data"].get("download", True),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,  # SSL에서는 batch size를 항상 일정하게 유지하는 게 안전
        persistent_workers=cfg["data"].get("persistent_workers", True),
    )
    return loader
