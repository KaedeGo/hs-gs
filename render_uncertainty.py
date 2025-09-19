#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import numpy as np
import csv

from gaussian_renderer import render, forward_k_times
from utils.image_utils import nll_kernel_density, ause_br, psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import HorseshoeModel

def render_set(dataset, name, iteration, views, gaussians, pipeline, background, depth_scale):
    psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    eval_depth = True if dataset.dataset_name == "LF" else False

    dataset_path, scene_name = os.path.split(dataset.model_path)

    render_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "gt")
    unc_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "uncertainty")
    unc_map_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "uncertainty_map")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(unc_path, exist_ok=True)
    makedirs(unc_map_path, exist_ok=True)

    if eval_depth:
        depth_render_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "depth_renders")
        depth_gts_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "depth_gt")
        depth_unc_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "depth_uncertainty")
        depth_unc_map_path = os.path.join(dataset_path, scene_name, name, "ours_{}".format(iteration), "depth_uncertainty_map")

        makedirs(depth_render_path, exist_ok=True)
        makedirs(depth_gts_path, exist_ok=True)
        makedirs(depth_unc_path, exist_ok=True)
        makedirs(depth_unc_map_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        out = forward_k_times(view, gaussians, pipeline, background, k=10) # TODO: choose k from [1, 2, 4, 8, 10, ...]
        rgbs = out['comp_rgbs'].detach()
        mean = out['comp_rgb'].detach()
        std = out['comp_std'].detach()

        mae = ((mean - gt)).abs()

        ause_mae, ause_err_mae, ause_err_by_var_mae = ause_br(std.reshape(-1), mae.reshape(-1), err_type='mae')
        mean_nll = nll_kernel_density(rgbs.permute(1,2,3,0), std, gt)

        psnr_all += psnr(mean, gt).mean().item()
        ssim_all += ssim(mean, gt).mean().item()
        lpips_all += lpips(mean, gt, net_type="vgg").mean().item()

        ause_mae_all += ause_mae.item()
        mean_nll_all += mean_nll.item()

        unc_vis_multiply = 10
        torchvision.utils.save_image(mean, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(unc_vis_multiply*std, os.path.join(unc_map_path, '{0:05d}'.format(idx) + ".png"))

        np.save(os.path.join(unc_path, '{0:05d}'.format(idx) + ".npy"), rgbs.cpu().numpy())

        if eval_depth: 
            depths = out['depths'].detach()
            depths = depths * depth_scale

            depth = depths.mean(dim=0)
            depth_std = depths.std(dim=0)
            depth_gt = view.depth

            depth_mae = ((depth - depth_gt)).abs()
            depth_ause_mae, depth_ause_err_mae, depth_ause_err_by_var_mae = ause_br(depth_std.reshape(-1), depth_mae.reshape(-1), err_type='mae')
            depth_ause_mae_all += depth_ause_mae

            torchvision.utils.save_image(depth, os.path.join(depth_render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth_gt, os.path.join(depth_gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth_std, os.path.join(depth_unc_map_path, '{0:05d}'.format(idx) + ".png"))

            np.save(os.path.join(depth_unc_path, '{0:05d}'.format(idx) + ".npy"), depths.cpu().numpy())

    psnr_all /= len(views)
    ause_mae_all /= len(views)
    mean_nll_all /= len(views)
    ssim_all /= len(views)
    lpips_all /= len(views)

    depth_ause_mae_all /= len(views)

    # csv_file = f"output/eval_results_{dataset.dataset_name}.csv"
    csv_file = f"{dataset_path}/eval_results.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if eval_depth: 
            results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all} Depth AUSE {depth_ause_mae_all}"
            print(results)
            writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all])
        else: 
            results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all}"
            print(results)
            writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all])


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = HorseshoeModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene.depth_scale)

        if not skip_test:
            render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene.depth_scale)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    # python render_uncertainty.py -m "output/best/LF/africa" --skip_train
    # python render_uncertainty.py -m "output/best/LF/basket" --skip_train
    # python render_uncertainty.py -m "output/best/LF/statue" --skip_train
    # python render_uncertainty.py -m "output/best/LF/torch" --skip_train