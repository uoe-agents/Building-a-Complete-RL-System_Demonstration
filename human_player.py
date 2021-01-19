import contextlib
import gym
import sys
import termios
import time

import utils


@contextlib.contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

def keyboard_code_to_action(keyboard_code: int) -> int:
    """
    Keyboard code transferred to action id (for Taxi-v3)

    :param keyboard_code (int): code of keyboard detected
    :return (int): action id for Taxi-v3 (and -1 for ESC)
    """
    if keyboard_code == 115:
        # S
        return 0
    elif keyboard_code == 119:
        # W
        return 1
    elif keyboard_code == 100:
        # D
        return 2
    elif keyboard_code == 97:
        # A
        return 3
    elif keyboard_code == 106:
        # J
        return 4
    elif keyboard_code == 107:
        # K
        return 5
    elif keyboard_code == 27:
        # ESC
        return -1
    else:
        raise ValueError(f"Unknown keyboard code {keyboard_code}!")

def get_keyboard_code():
    with raw_mode(sys.stdin):
        try:
            ch = sys.stdin.read(1)
            if ch and ch != chr(4):
                return keyboard_code_to_action(ord(ch))
        except EOFError:
            pass

def human_player(env):
    """
    Play Taxi-v3 as a human player with WASD keys + J (Pickup) K (Dropoff)
    """
    print("Use WASD to move in the environment, use J and K to Pickup or Dropoff the passenger and end game with STOP or keyboard interrupt (Ctrl-C)")
    env.reset()
    ep_return = 0
    env.render()

    while True:
        input = get_keyboard_code()
        if input == -1:
            return
        _, rew, done, _ = env.step(input)
        ep_return += rew
        env.render()
        if done:
            if ep_return > 0:
                print(f"EPISODE FINISHED WITH RETURN {ep_return} - SOLVED")
            else:
                print(f"EPISODE FINISHED WITH RETURN {ep_return} - FAILED")
            env.reset()
            ep_return = 0
            env.render()
    return


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    human_player(env)
    env.close()
