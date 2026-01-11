#!/usr/bin/env python3
from time import sleep
from pidog import Pidog


def main():
    dog = Pidog()
    sleep(0.5)
    # Small, clamped moves to verify axes without slamming.
    steps = [
        ([0, 0, 0], "center"),
        ([10, 0, 0], "yaw +10"),
        ([-10, 0, 0], "yaw -10"),
        ([0, 10, 0], "roll +10"),
        ([0, -10, 0], "roll -10"),
        ([0, 0, 10], "pitch +10"),
        ([0, 0, -10], "pitch -10"),
        ([0, 0, 0], "center"),
    ]
    for angles, label in steps:
        print(f"Move: {label} -> {angles}")
        dog.head_move_raw([angles], speed=40)
        dog.wait_head_done()
        sleep(0.8)
    dog.close()


if __name__ == "__main__":
    main()
