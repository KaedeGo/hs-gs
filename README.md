### 1.Dataset path
```
/home/fwu/Datasets/3DGS  # root path
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
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

### webGL viewer
```
https://github.com/antimatter15/splat 
download the .ply file and cameraa.json from output directory and drag it to website
```