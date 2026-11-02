"""Lossless PNG cache of resized RDD2022 images, mirroring image_path so PotholeDataset reads it unchanged.

The resize is the same T.Resize((size, size)) used by build_transforms, so a cached image fed through the
evaluation transform gives exactly the same tensor as the original JPG.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from src.data.dataset import DATASET_ROOT, load_split_frame


def cache_root(size, dataset_root=DATASET_ROOT):
    return Path(dataset_root).parent / f"RDD2022_{size}"


def safe_relative(relative):
    """Reject absolute paths, drives and parent references before joining with a root."""
    windows = PureWindowsPath(relative)
    if windows.drive or windows.root or ".." in windows.parts:
        raise ValueError(f"unsafe image_path: {relative!r}")
    return PurePosixPath(relative)


def cache_path(relative):
    """image_path with .png instead of .jpg."""
    return safe_relative(relative).with_suffix(".png").as_posix()


def cached_frame(frame):
    """Copy of the frame pointing image_path to the cached PNGs."""
    return frame.assign(image_path=frame.image_path.map(cache_path))


def inside(root, relative):
    # chequeo lexico a proposito: resolve() en windows da rutas inconsistentes con el junction
    # mientras otros hilos crean carpetas, y tiraba falsos positivos
    root = Path(os.path.normpath(root))
    target = Path(os.path.normpath(root / safe_relative(relative)))
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes {root}: {relative!r}")
    return target


def cache_one(relative, src_root, dst_root, resize):
    """Write one PNG atomically; returns False when it already exists."""
    target = inside(dst_root, cache_path(relative))
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(inside(src_root, relative)) as image:
        resized = resize(image.convert("RGB"))
    # primero a un temporal, asi un corte no deja un png a medias
    temp = target.with_name(target.name + ".tmp")
    resized.save(temp, format="PNG")
    os.replace(temp, target)
    return True


def build_cache(frame, src_root, dst_root, size, workers=4):
    """Cache every image of the frame; idempotent, returns how many files were written."""
    resize = T.Resize((size, size))
    paths = frame.image_path.tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda relative: cache_one(relative, src_root, dst_root, resize), paths)
        return sum(tqdm(results, total=len(paths), desc=f"cache {size}px", mininterval=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    frame = load_split_frame()
    target = cache_root(args.size)
    start = time.time()
    written = build_cache(frame, DATASET_ROOT, target, args.size, args.workers)
    print(f"{written} written, {len(frame) - written} already cached, {time.time() - start:.0f}s -> {target}")


if __name__ == "__main__":
    main()
