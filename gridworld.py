#!/usr/bin/env python3

import argparse
import envs
from gym_minigrid.window import Window
import gym


parser = argparse.ArgumentParser()

parser.add_argument('-m', '--manual',action='store_true',
                    dest='manual', help='Manual control mode for agent')

parser.add_argument('-e', '--env', dest='env', help="gym environment to load",
                    default='MiniGrid-MDP-AIMAGrid-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)


args = parser.parse_args()

def redraw(observation):
    img = env.render('rgb_array', tile_size=args.tile_size)
    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, done, _ = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.up)
        return
    if event.key == 'down':
        step(env.actions.down)
        return



if args.manual == True:
    env = gym.make(args.env)    
    window = Window('gym_minigrid - ' + args.env)
    window.reg_key_handler(key_handler)
    reset()

    # Blocking event loop
    window.show(block=True)
else:

    raise Exception("Auto mode not implemented yet")