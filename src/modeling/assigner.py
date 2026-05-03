import torch


def locations_for_features(features, strides, device):
    all_locations = []
    all_strides = []
    for feat, stride in zip(features, strides):
        h, w = feat.shape[-2:]
        shifts_x = (torch.arange(w, device=device) + 0.5) * stride
        shifts_y = (torch.arange(h, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        locations = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        all_locations.append(locations)
        all_strides.append(torch.full((locations.shape[0],), stride, device=device))
    return torch.cat(all_locations, dim=0), torch.cat(all_strides, dim=0)


def assign_fcos_targets(locations, strides, targets, num_classes=10, center_radius=1.5):
    labels_out, reg_out, center_out = [], [], []
    inf = torch.tensor(1e8, device=locations.device)
    ranges = {
        8: (0, 96),
        16: (64, 192),
        32: (128, inf),
    }
    for target in targets:
        boxes = target["boxes"].to(locations.device)
        labels = target["labels"].to(locations.device)
        n = locations.shape[0]
        labels_per = torch.full((n,), num_classes, dtype=torch.long, device=locations.device)
        reg_per = torch.zeros((n, 4), dtype=torch.float32, device=locations.device)
        center_per = torch.zeros((n,), dtype=torch.float32, device=locations.device)
        if len(boxes) == 0:
            labels_out.append(labels_per)
            reg_out.append(reg_per)
            center_out.append(center_per)
            continue

        xs, ys = locations[:, 0], locations[:, 1]
        l = xs[:, None] - boxes[:, 0][None]
        t = ys[:, None] - boxes[:, 1][None]
        r = boxes[:, 2][None] - xs[:, None]
        b = boxes[:, 3][None] - ys[:, None]
        reg = torch.stack([l, t, r, b], dim=2)
        inside_box = reg.min(dim=2).values > 0

        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        radius = strides[:, None] * center_radius
        center_box = torch.stack([cx[None] - radius, cy[None] - radius, cx[None] + radius, cy[None] + radius], dim=2)
        cb_l = xs[:, None] - center_box[:, :, 0]
        cb_t = ys[:, None] - center_box[:, :, 1]
        cb_r = center_box[:, :, 2] - xs[:, None]
        cb_b = center_box[:, :, 3] - ys[:, None]
        inside_center = torch.stack([cb_l, cb_t, cb_r, cb_b], dim=2).min(dim=2).values > 0

        max_reg = reg.max(dim=2).values
        in_level = torch.zeros_like(inside_box)
        for stride, (lo, hi) in ranges.items():
            mask = strides == stride
            in_level[mask] = (max_reg[mask] >= lo) & (max_reg[mask] <= hi)

        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[None].repeat(n, 1)
        areas[~(inside_box & inside_center & in_level)] = inf
        min_area, matched = areas.min(dim=1)
        pos = min_area < inf
        if pos.any():
            labels_per[pos] = labels[matched[pos]]
            reg_per[pos] = reg[torch.arange(n, device=locations.device)[pos], matched[pos]]
            lr = reg_per[pos][:, [0, 2]]
            tb = reg_per[pos][:, [1, 3]]
            center_per[pos] = torch.sqrt(
                (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-6))
                * (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-6))
            )
        labels_out.append(labels_per)
        reg_out.append(reg_per)
        center_out.append(center_per)
    return torch.stack(labels_out), torch.stack(reg_out), torch.stack(center_out)

