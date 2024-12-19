### 1.Dataset path
```
/home/fwu/Datasets/3DGS  # root path
/home/fwu/Datasets/3DGS/360_v2 # path for Mip-Nerf360 dataset(bicycle  bonsai  counter  flowers  garden  kitchen  room  stump  treehill)
/home/fwu/Datasets/3DGS/tandt_db/tandt # path for Tanks&Temple dataset(train  truck)
/home/fwu/Datasets/3DGS/tandt_db/db # path for Deep Blending dataset(drjohnson  playroom)
/home/fwu/Datasets/3DGS/nerf_synthetic # path for Synthetic NeRF(chair  drums  ficus  hotdog  lego  materials  mic ship) colmap needed for sparse point cloud
```

### 2. environment on CU12.4
```
conda env create --file environment_cu12.4.yml
conda activate gs_cu12
```

### 3.run initial GS
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split

python train.py -s home/fwu/Datasets/3DGS/tandt_db/db/drjohnson --eval --disable_viewer #example

python render.py -m <path to trained model> # Generate renderings

python render.py -m output/ddb928c4-5/ #example

python metrics.py -m <path to trained model> # Compute error metrics on renderings

python metrics.py -m output/ddb928c4-5/ # example
  SSIM :    0.9026289
  PSNR :   29.4206161
  LPIPS:    0.2388636
```

### webGL viewer
```
https://github.com/antimatter15/splat 
download the .ply file and cameraa.json from output directory and drag it to website
```