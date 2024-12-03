
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
        "--weights", 
        type=str, 
        default="./weights", 
        help="folder to save weights"
    )
    parser.add_argument(
        "--weights_name", 
        type=str, 
        default="model.pt", 
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default=''
    )
    parser.add_argument(
        '--preprocess', 
       action='store_true'
    )    
    parser.add_argument(
        '--dig_sep', 
       action='store_true'
    )



    return parser.parse_args()
