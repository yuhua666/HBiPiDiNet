# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet_REACT',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        conv_cfg = dict(type='BiConv'),
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN_REACT_3x3',
        in_channels=[64, 128, 256, 512],
        conv_cfg = dict(type='BiConv'),
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='RFGenerator', 
            fpn_layer='p3', 
            fraction=0.5, 
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0)))
# training and testing settings
train_cfg=dict(
    assigner=dict(
        type='HieAssigner', # Hierarchical Label Assigner (HLA)
        ignore_iof_thr=-1,
        gpu_assign_thr=512,
        iou_calculator=dict(type='BboxDistanceMetric'),
        assign_metric='kl', 
        topk=[3,1],
        ratio=0.9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg=dict(
    nms_pre=3000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=3000)

# optimizer
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    # policy='multi',
    warmup='linear',
    warmup_iters=1250,
    warmup_ratio=0.001,
    step=[8, 11])

total_epochs = 12  
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12, metric='bbox')

#dataset
dataset_type = 'AITODDataset'
data_root = '/root/autodl-tmp/AI-TOD/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/aitod_trainval_v1.json',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/aitod_test_v1.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/aitod_test_v1.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))
#evaluation = dict(interval=12, metric='bbox')