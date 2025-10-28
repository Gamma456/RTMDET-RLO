# # Copyright (c) OpenMMLab. All rights reserved.
# import argparse
# import os
# from typing import Sequence

# import mmcv
# from mmdet.apis import inference_detector, init_detector
# from mmengine import Config, DictAction
# from mmengine.registry import init_default_scope
# from mmengine.utils import ProgressBar

# from mmyolo.registry import VISUALIZERS
# from mmyolo.utils.misc import auto_arrange_images, get_file_list
# import numpy as np
# from mmdet.apis import imshow_det_bboxes


# def parse_args():
#     parser = argparse.ArgumentParser(description='Visualize feature map')
#     parser.add_argument(
#         '--img', default='data/train_data/test/images', help='Image path, include image file, dir and URL.')
#     parser.add_argument('--config', default=r'C:\Users\Owner\Desktop\RTMDet_RLO\run\2025_final\p6_34\RBD_k_size=53_stage_234_D_boundary_norm\weight_bbox=2.5,weight_cls=0.1\lr=0.001\500epoches_no_spp\rotated_rtmdet_tiny-3x-dota.py', help='Config file')
#     parser.add_argument('--checkpoint', default=r'C:\Users\Owner\Desktop\RTMDet_RLO\run\2025_final\p6_34\RBD_k_size=53_stage_234_D_boundary_norm\weight_bbox=2.5,weight_cls=0.1\lr=0.001\500epoches_no_spp\epoch_500.pth', help='Checkpoint file')
#     parser.add_argument(
#         '--out-dir', default='work_dirs/final/feature_map', help='Path to output file')
#     parser.add_argument(
#         '--target-layers',
#         default=['bbox_head.rtm_ang[0]'],
#         nargs='+',
#         type=str,
#         help='The target layers to get feature map, if not set, the tool will '
#         'specify the backbone')
#     parser.add_argument(
#         '--preview-model',
#         default=False,
#         action='store_true',
#         help='To preview all the model layers')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--score-thr', type=float, default=0.03, help='Bbox score threshold')
#     parser.add_argument(
#         '--show', action='store_true', help='Show the featmap results')
#     parser.add_argument(
#         '--channel-reduction',
#         default='select_max',
#         help='Reduce multiple channels to a single channel')
#     parser.add_argument(
#         '--topk',
#         type=int,
#         default=4,
#         help='Select topk channel to show by the sum of each channel')
#     parser.add_argument(
#         '--arrangement',
#         nargs='+',
#         type=int,
#         default=[2, 2],
#         help='The arrangement of featmap when channel_reduction is '
#         'not None and topk > 0')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     args = parser.parse_args()
#     return args


# class ActivationsWrapper:

#     def __init__(self, model, target_layers):
#         self.model = model
#         self.activations = []
#         self.handles = []
#         self.image = None
#         for target_layer in target_layers:
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_activation))

#     def save_activation(self, module, input, output):
#         self.activations.append(output)

#     def __call__(self, img_path):
#         self.activations = []
#         results = inference_detector(self.model, img_path)
#         return results, self.activations

#     def release(self):
#         for handle in self.handles:
#             handle.remove()


# def main():
#     args = parse_args()

#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)

#     init_default_scope(cfg.get('default_scope', 'mmyolo'))

#     channel_reduction = args.channel_reduction
#     if channel_reduction == 'None':
#         channel_reduction = None
#     assert len(args.arrangement) == 2

#     model = init_detector(args.config, args.checkpoint, device=args.device)

#     if not os.path.exists(args.out_dir) and not args.show:
#         os.mkdir(args.out_dir)

#     if args.preview_model:
#         print(model)
#         print('\n This flag is only show model, if you want to continue, '
#               'please remove `--preview-model` to get the feature map.')
#         out_path = os.path.join(os.getcwd(), 'work_dirs/final/feature_map/model_struct.txt')
#         with open(out_path, 'w', encoding='utf-8') as f:
#           f.write('=== Model repr ===\n')
#           f.write(str(model) + '\n\n')
#         print(f'Model structure saved to {out_path}')
#         return

#     target_layers = []
#     for target_layer in args.target_layers:
#         try:
#             target_layers.append(eval(f'model.{target_layer}'))
#         except Exception as e:
#             print(model)
#             raise RuntimeError('layer does not exist', e)

#     activations_wrapper = ActivationsWrapper(model, target_layers)

#     # init visualizer
#     visualizer = VISUALIZERS.build(model.cfg.visualizer)
#     visualizer.dataset_meta = model.dataset_meta

#     # get file list
#     image_list, source_type = get_file_list(args.img)

#     progress_bar = ProgressBar(len(image_list))
#     for image_path in image_list:
#         result, featmaps = activations_wrapper(image_path)
#         if not isinstance(featmaps, Sequence):
#             featmaps = [featmaps]

#         flatten_featmaps = []
#         for featmap in featmaps:
#             if isinstance(featmap, Sequence):
#                 flatten_featmaps.extend(featmap)
#             else:
#                 flatten_featmaps.append(featmap)

#         img = mmcv.imread(image_path)
#         img = mmcv.imconvert(img, 'bgr', 'rgb')

#         if source_type['is_dir']:
#             filename = os.path.relpath(image_path, args.img).replace('/', '_')
#         else:
#             filename = os.path.basename(image_path)
#         out_file = None if args.show else os.path.join(args.out_dir, filename)

#         # show the results
#         shown_imgs = []
#         visualizer.add_datasample(
#             'result',
#             img,
#             data_sample=result,
#             draw_gt=False,
#             show=False,
#             wait_time=0,
#             out_file=None,
#             pred_score_thr=args.score_thr,
#             draw_bbox_kwargs=dict(line_widths=5))
#         drawn_img = visualizer.get_image()

#         for featmap in flatten_featmaps:
#             shown_img = visualizer.draw_featmap(
#                 featmap[0],
#                 drawn_img,
#                 channel_reduction=channel_reduction,
#                 topk=args.topk,
#                 arrangement=args.arrangement)
#             shown_imgs.append(shown_img)

#         shown_imgs = auto_arrange_images(shown_imgs)

#         progress_bar.update()
#         if out_file:
#             mmcv.imwrite(shown_imgs[..., ::-1], out_file)

#         if args.show:
#             visualizer.show(shown_imgs)

#     if not args.show:
#         print(f'All done!'
#               f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


# # Please refer to the usage tutorial:
# # https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/visualization.md # noqa
# if __name__ == '__main__':
#     main()







# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from typing import Sequence

import cv2
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmyolo.registry import VISUALIZERS
from mmyolo.utils.misc import auto_arrange_images, get_file_list


# ### 以下做了修改繪製pred_box的方式 使線條粗細跟顏色可以調整 ###

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature map')
    parser.add_argument(
        '--img', default='data/train_data/test/images', help='Image path, include image file, dir and URL.')
    parser.add_argument('--config', default=r'C:\Users\Owner\Desktop\RTMDet_RLO\run\2025_final\p6_34\RBD_k_size=53_stage_234_D_boundary_norm\weight_bbox=2.5,weight_cls=0.1\lr=0.001\500epoches_no_spp\rotated_rtmdet_tiny-3x-dota.py', help='Config file')
    parser.add_argument('--checkpoint', default=r'C:\Users\Owner\Desktop\RTMDet_RLO\run\2025_final\p6_34\RBD_k_size=53_stage_234_D_boundary_norm\weight_bbox=2.5,weight_cls=0.1\lr=0.001\500epoches_no_spp\epoch_500.pth', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='work_dirs/final/feature_map/best', help='Path to output file')
    parser.add_argument(
        '--target-layers', nargs='+', default=['bbox_head.rtm_reg[0]'],
        help='List of model layers to hook for feature maps')
    parser.add_argument(
        '--preview-model', action='store_true', default=False,
        help='Print model structure and exit')
    parser.add_argument(
        '--device', default='cuda:0',
        help='Computation device')
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='Score threshold for predicted boxes')
    parser.add_argument(
        '--show', action='store_true', default=False,
        help='Show results in window instead of saving')
    parser.add_argument(
        '--bbox-thickness', type=int, default=8,
        help='Thickness of predicted bounding boxes')
    parser.add_argument(
        '--draw-pred', dest='draw_pred', action='store_true',
        help='Enable drawing predicted boxes')
    parser.add_argument(
        '--no-draw-pred', dest='draw_pred', action='store_false',
        help='Disable drawing predicted boxes')
    parser.set_defaults(draw_pred=False)
    parser.add_argument(
        '--channel-reduction', default='select_max',
        help='Channel reduction method for feature maps')
    parser.add_argument(
        '--topk', type=int, default=4,
        help='Number of top channels to display')
    parser.add_argument(
        '--arrangement', nargs=2, type=int, default=[2, 2],
        help='Grid arrangement (rows, cols) for feature maps')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Overwrite config options')
    return parser.parse_args()


class ActivationsWrapper:
    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        for layer in target_layers:
            self.handles.append(layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        self.activations = []
        results = inference_detector(self.model, img_path)
        return results, self.activations

    def release(self):
        for h in self.handles:
            h.remove()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    # Initialize model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if args.preview_model:
        # Print and save model structure
        print(model)
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'model_struct.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('=== Model repr ===\n')
            f.write(str(model) + '\n')
        print(f'Model structure saved to {out_path}')
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Hook target layers
    target_modules = []
    for name in args.target_layers:
        try:
            target_modules.append(eval(f'model.{name}'))
        except Exception:
            raise RuntimeError(f'Layer not found: {name}')
    hook = ActivationsWrapper(model, target_modules)

    # Build visualizer
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # Gather input images
    image_list, src_type = get_file_list(args.img)
    prog = ProgressBar(len(image_list))

    for img_path in image_list:
        # Inference + feature hook
        result, feats = hook(img_path)

        # Extract bboxes from DetDataSample
        if isinstance(result, DetDataSample) and hasattr(result, 'pred_instances'):
            inst = result.pred_instances
            bboxes = inst.bboxes.cpu().numpy()
            scores = inst.scores.cpu().numpy()
        else:
            raise RuntimeError('Unexpected result type, please update code')

        # Flatten feature maps
        if not isinstance(feats, Sequence):
            feats = [feats]
        flat_feats = []
        for fm in feats:
            if isinstance(fm, Sequence):
                flat_feats.extend(fm)
            else:
                flat_feats.append(fm)

        # Read image
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        # Prepare base image with optional predicted boxes
        if args.draw_pred:
            drawn = img.copy()
            for bbox, score in zip(bboxes, scores):
                if score < args.score_thr:
                    continue
                # Rotated bbox: [xc, yc, w, h, angle]
                if len(bbox) == 5:
                    xc, yc, w, h, angle = bbox
                    rect = ((float(xc), float(yc)), (float(w), float(h)), float(angle))
                    pts = cv2.boxPoints(rect).astype(int)
                    cv2.polylines(drawn, [pts], isClosed=True, color=(255, 0, 0), thickness=args.bbox_thickness)
                else:
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), (255, 0, 0), thickness=args.bbox_thickness)
        else:
            drawn = img

        # Visualize feature maps overlayed on drawn image
        shown_imgs = []
        for fm in flat_feats:
            shown_imgs.append(
                visualizer.draw_featmap(
                    fm[0], drawn,
                    channel_reduction=args.channel_reduction,
                    topk=args.topk,
                    arrangement=tuple(args.arrangement)
                )
            )
        grid = auto_arrange_images(shown_imgs, image_column=args.arrangement[1])

        # Save or show
        if src_type['is_dir']:
            fname = os.path.relpath(img_path, args.img).replace('/', '_')
        else:
            fname = os.path.basename(img_path)
        out_file = os.path.join(args.out_dir, fname)
        mmcv.imwrite(grid[..., ::-1], out_file)
        if args.show:
            visualizer.show(grid)

        prog.update()

    hook.release()
    print(f'All done! Results saved to {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
