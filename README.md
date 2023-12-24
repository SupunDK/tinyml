
## Requirements

- numpy
- torch
- tqdm
- torchvision
- argparse


## Training and Evaluation

Go to the `training_python` folder.

    python train.py --tqdm_ # show progress bar
    python train.py --ensemble # define multiple threshold  points as ensembles

use --mode as train, test or total for required mode
use `--factor 2.079 --threshold 9.211` for the optimum result (This is the default)

## Uploading to the board

The solution does not use CubeMX. You can directly compile the uVision project and upload to the board.

