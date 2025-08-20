# Entry point to run training or evaluation
import argparse
import train
import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--img_size", type=int, default=720)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "train":
        train.run(args)
    else:
        evaluate.run(args)
