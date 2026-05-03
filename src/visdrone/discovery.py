from pathlib import Path


def _find_split(root, pattern, image_child, ann_child=None):
    root = Path(root).resolve()
    for split_dir in root.rglob(pattern):
        image_dir = split_dir / image_child
        ann_dir = split_dir / ann_child if ann_child else None
        if image_dir.is_dir() and (ann_dir is None or ann_dir.is_dir()):
            return split_dir, image_dir, ann_dir
    return None


def find_det_split(data_root=".", split="train"):
    name = f"VisDrone2019-DET-{split}"
    found = _find_split(data_root, name, "images", "annotations")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find {name}/images and {name}/annotations under {Path(data_root).resolve()}"
    )


def find_vid_split(data_root=".", split="val"):
    name = f"VisDrone2019-VID-{split}"
    found = _find_split(data_root, name, "sequences", "annotations")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find {name}/sequences and {name}/annotations under {Path(data_root).resolve()}"
    )


def summarize_visdrone(data_root="."):
    summary = {}
    for split in ("train", "val"):
        try:
            _, image_dir, ann_dir = find_det_split(data_root, split)
            summary[f"det_{split}"] = {
                "images": len([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]),
                "annotations": len(list(ann_dir.glob("*.txt"))),
                "image_dir": str(image_dir),
                "annotation_dir": str(ann_dir),
            }
        except FileNotFoundError:
            summary[f"det_{split}"] = None
    for split in ("train", "val", "test-dev", "test-challenge"):
        try:
            _, seq_dir, ann_dir = find_vid_split(data_root, split)
            summary[f"vid_{split}"] = {
                "sequences": len([p for p in seq_dir.iterdir() if p.is_dir()]),
                "annotations": len(list(ann_dir.glob("*.txt"))) if ann_dir else 0,
                "sequence_dir": str(seq_dir),
                "annotation_dir": str(ann_dir) if ann_dir else None,
            }
        except FileNotFoundError:
            summary[f"vid_{split}"] = None
    return summary

