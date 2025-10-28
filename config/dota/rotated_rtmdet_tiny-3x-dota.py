_base_ = './rotated_rtmdet_l-3x-dota.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=( 5,),
        # out_indices=( 3 ,4 , 5),
        # out_indices=(2, 3 , 4),
        # init_cfg=dict(
        #     type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
        init_cfg=None),        
    # neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),  #p5_234
    # neck=dict(in_channels=[ 384], out_channels=384, num_csp_blocks=1),      #p5
    # neck=dict(in_channels=[192, 288, 384], out_channels=192, num_csp_blocks=1),  #p6_345
    neck=dict(in_channels=[ 384 ], out_channels=384, num_csp_blocks=1),      #p6
    bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[64]),#8 16 32
        exp_on_reg=False,
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.5),
    ))

# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=8)
