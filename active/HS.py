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
        Real-world version: Selects views by maximizing the integrated predictive uncertainty. Does not use ground truth.
        """
        uncertainty_scores = torch.zeros(len(candidate_cameras))

        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating uncertainty on candidate views")):

            out = forward_k_times(cam, gaussians, pipe, background, n_samples=10) # 建议增加 n_samples 以获得更稳定的不确定性估计
            
            std = out['comp_std'].detach()

            score = torch.sum(std**2) 
            
            uncertainty_scores[idx] = score

        
        print(f"Uncertainty scores: {uncertainty_scores.tolist()}")

        _, indices = torch.sort(uncertainty_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        return selected_idxs