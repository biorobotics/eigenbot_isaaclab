import pygame
import numpy as np
import time


def init_joystick():
    pygame.init()
    pygame.joystick.init()
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(joy.get_name())
    return joy


def read(joy):
    pygame.event.get()
    axes = []
    buttons = []
    povs = []
    for ax in range(joy.get_numaxes()):
        axes.append(joy.get_axis(ax))
    for button in range(joy.get_numbuttons()):
        buttons.append(joy.get_button(button))
    for hat in range(joy.get_numhats()):
        povs.append(joy.get_hat(hat))
    return axes, buttons, povs


def test():
    pygame.init()
    pygame.joystick.init()
    pygame.joystick.Joystick(0).init()

    while True:
        axes, buttons, povs = read(pygame.joystick.Joystick(0))
        print(np.round(axes, 2), buttons, povs)
        time.sleep(1)


if __name__ == '__main__':
    joy = init_joystick()
    time_start = time.time()
    while time.time() < time_start + 10:
        axes, buttons, povs = read(joy)
        print(np.round(axes, 2), buttons, povs)
        print(np.arctan2(-axes[0], -axes[1]))
        time.sleep(0.2)
