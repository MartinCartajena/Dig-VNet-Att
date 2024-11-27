
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Training vNET model for segmentation of lung CT"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--loss", 
        type=str, 
        default="softdice", 
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis_images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis_freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default="./weights", 
        help="folder to save weights"
    )
    parser.add_argument(
        "--logs", 
        type=str, 
        default="./logs", 
        help="folder to save logs"
    )
    parser.add_argument(
        "--aug_scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug_angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default=''
    )
    parser.add_argument(
        '--dig_sep', 
       action='store_true'
    )



    return parser.parse_args()
