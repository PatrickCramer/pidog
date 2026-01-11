#!/usr/bin/env python3
from time import sleep
from robot_hat import Servo, utils


def main():
    # Directly test the head roll servo (pin 6) with small moves.
    utils.reset_mcu()
    sleep(0.5)
    roll_servo = Servo(6)
    for angle in (0, -10, 0, 10, 0):
        roll_servo.angle(angle)
        sleep(1.0)


if __name__ == "__main__":
    main()
