import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import forward_k_times
from scene import Scene
from utils.image_utils import ause_br


class HSSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
    
    def nbvs(self, gaussians, scene: Scene, pipe, background) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))

        viewpoint_cams = scene.getTrainCameras().copy()

        candidate_cameras = scene.getCandidateCameras()
        # Run heesian on training set
        return self.select_single_view(candidate_cameras, candidate_views, gaussians, pipe, background)

    
    def forward(self, x):
        return x
    
    
    def select_single_view(self, candidate_cameras, candidate_views, gaussians, pipe, background, num_views=1):
        """
        A memory effcient way when doing single view selection
        """
        ause_scores = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating uncertainty on candidate views")):

            out = forward_k_times(cam, gaussians, pipe, background, n_samples=2)
            gt = cam.original_image[0:3, :, :]
            mean = out['comp_rgb'].detach()
            std = out['comp_std'].detach()
            mae = ((mean - gt)).abs()

            ause_mae, ause_err_mae, ause_err_by_var_mae = ause_br(std.reshape(-1), mae.reshape(-1), err_type='mae')

            ause_scores[idx] = ause_mae

        
        print(f"ause_scores: {ause_scores.tolist()}")

        _, indices = torch.sort(ause_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        return selected_idxs
