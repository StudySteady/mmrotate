import os
import numpy as np
import mmcv
from mmcv import Config

import mmrotate.datasets  # noqa: F401
import mmrotate.models    # noqa: F401

from mmdet.datasets import build_dataset
from mmrotate.datasets.dota import DOTADataset


CONFIG = 'configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'
RESULT_PKL = 'trainval_results_21046.pkl'


def to_bbox_array(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)

    if isinstance(x, list):
        if len(x) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.array(x, dtype=np.float32)
        return arr.reshape(-1, 5)

    return np.zeros((0, 5), dtype=np.float32)


def to_label_array(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.int64)

    if isinstance(x, list):
        if len(x) == 0:
            return np.array([], dtype=np.int64)
        return np.array(x, dtype=np.int64)

    return np.array([], dtype=np.int64)


_original_load_annotations = DOTADataset.load_annotations


def patched_load_annotations(self, ann_folder):
    data_infos = _original_load_annotations(self, ann_folder)

    for info in data_infos:
        ann = info.get('ann', {})
        ann['bboxes'] = to_bbox_array(ann.get('bboxes', []))
        ann['labels'] = to_label_array(ann.get('labels', []))
        ann['bboxes_ignore'] = to_bbox_array(ann.get('bboxes_ignore', []))
        ann['labels_ignore'] = to_label_array(ann.get('labels_ignore', []))
        info['ann'] = ann

    print(f'[patched_load_annotations] total samples: {len(data_infos)}')
    return data_infos


DOTADataset.load_annotations = patched_load_annotations


def main():
    cfg = Config.fromfile(CONFIG)

    cfg.data.test.ann_file = 'data/split_ss_dota/trainval/annfiles/'
    cfg.data.test.img_prefix = 'data/split_ss_dota/trainval/images/'
    cfg.data.test.test_mode = True
    # cfg.data.test.filter_empty_gt = False

    print('ann_file =', cfg.data.test.ann_file)
    print('img_prefix =', cfg.data.test.img_prefix)
    print('RESULT_PKL =', RESULT_PKL)
    print('RESULT_PKL exists =', os.path.exists(RESULT_PKL))

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(RESULT_PKL)

    print(f'dataset size: {len(dataset)}')
    print(f'outputs size: {len(outputs)}')

    assert len(dataset) == len(outputs), (
        f'长度不一致: dataset={len(dataset)}, outputs={len(outputs)}'
    )

    eval_kwargs = dict(metric='mAP')
    if hasattr(cfg, 'evaluation') and isinstance(cfg.evaluation, dict):
        for k, v in cfg.evaluation.items():
            if k not in [
                'interval', 'tmpdir', 'start', 'gpu_collect',
                'save_best', 'rule', 'dynamic_intervals'
            ]:
                eval_kwargs[k] = v

    print('Start evaluating from existing results.pkl ...')
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print('Evaluation result:')
    print(metric)


if __name__ == '__main__':
    main()