log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-07, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=1200)
checkpoint_config = dict(interval=100)
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe'),
    neck=dict(
        type='FPNC', in_channels=[64, 128, 256, 512], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'IcdarDataset'
data_root = 'data/icdar2015'
train = dict(
    type='IcdarDataset',
    ann_file='data/icdar2015/instances_training.json',
    img_prefix='data/icdar2015/imgs',
    pipeline=None)
test = dict(
    type='IcdarDataset',
    ann_file='data/icdar2015/instances_test.json',
    img_prefix='data/icdar2015/imgs',
    pipeline=None)
train_list = [
    dict(
        type='IcdarDataset',
        ann_file='data/icdar2015/instances_training.json',
        img_prefix='data/icdar2015/imgs',
        pipeline=None)
]
test_list = [
    dict(
        type='IcdarDataset',
        ann_file='data/icdar2015/instances_test.json',
        img_prefix='data/icdar2015/imgs',
        pipeline=None)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline_r18 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline_1333_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 736),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
img_norm_cfg_r50dcnv2 = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline_r50dcnv2 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[122.67891434, 116.66876762, 104.00698793],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.67891434, 116.66876762, 104.00698793],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file='data/icdar2015/instances_training.json',
                img_prefix='data/icdar2015/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='ImgAug',
                args=[['Fliplr', 0.5], {
                    'cls': 'Affine',
                    'rotate': [-10, 10]
                }, ['Resize', [0.5, 3.0]]]),
            dict(type='EastRandomCrop', target_size=(640, 640)),
            dict(type='DBNetTargets', shrink_ratio=0.4),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
                visualize=dict(flag=False, boundary_key='gt_shrink')),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr',
                    'gt_thr_mask'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file='data/icdar2015/instances_test.json',
                img_prefix='data/icdar2015/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 736),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file='data/icdar2015/instances_test.json',
                img_prefix='data/icdar2015/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 736),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=100, metric='hmean-iou')
