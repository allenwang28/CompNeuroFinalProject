import argparse
import sys
import pickle

import numpy as np
import pygame
from pygame.color import THECOLORS
from pygame.locals import QUIT

pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()


def render_trajectory(trajectory, rescale):
  for pt in trajectory:
    for event in pygame.event.get():
      if event.type == QUIT:
        sys.exit(0)
    screen.fill(THECOLORS['white'])
    if rescale:
      pt = (int(pt[0] * 600), int(pt[1] * 600))
    pt = (int(pt[0]), int(pt[1]))
    pygame.draw.circle(screen, THECOLORS['red'], pt, 25)
    clock.tick(50)
    pygame.display.flip()
    

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('fname', type=str, help='path to file containing trajectories')
  parser.add_argument('--show_n', type=int, help='number of trajectories to show (default all)')
  parser.add_argument('--rescale', action='store_true', help='rescale from predictions from rnn')
  args = parser.parse_args()
  with open(args.fname, 'rb') as f:
    trajectories = pickle.load(f)
  for trajectory in trajectories:
    render_trajectory(trajectory, args.rescale)

