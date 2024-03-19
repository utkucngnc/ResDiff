from src.SimpleSR.train import main
import argparse
import src.SimpleSR.config as config
"""
parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
parser.add_argument("--model_arch_name",
                    type=str,
                    default=config.model_arch_name)
parser.add_argument("--upscale_factor",
                    type=int,
                    default=config.upscale_factor)
parser.add_argument("--inputs_path",
                    type=str,
                    default="./data_128/678.png",
                    help="Low-resolution image path.")
parser.add_argument("--output_path",
                    type=str,
                    default="./678.png",
                    help="Super-resolution image path.")
parser.add_argument("--model_weights_path",
                    type=str,
                    default=f"./results/{config.exp_name}/g_best.pth.tar",
                    help="Model weights file path.")
parser.add_argument("--device_type",
                    type=str,
                    default="cuda",
                    choices=["cpu", "cuda"])
args = parser.parse_args()

main(args)
"""

main()