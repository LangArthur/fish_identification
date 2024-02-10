#!/usr/bin/env python3
import argparse
import tensorflow as tf
import keras;

def infer():
    print("infer")

def train():
    keras.models.load_model("")

def main():
    print(tf.version.VERSION)
    parser = argparse.ArgumentParser(description="Fish identification")
    # add flag gestion
    parser.add_argument("-t", "--train", help="train mode", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    else:
        infer()

if __name__ == '__main__':
    main()
