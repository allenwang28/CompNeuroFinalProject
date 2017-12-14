import argparse
import random

import progressbar
import numpy as np
import pickle
import pymunk


BALL_M = 10
BALL_R = 25
GRAVITY = (0, 900.)
SCREEN_SIZE = 600 
ELASTICITY = 1 - 1e-5
FRICTION = 0.0 


def cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None, help='random number generator seed')
  parser.add_argument('--n_trajectories', type=int, default=1, help='number of trajectories to simulate')
  parser.add_argument('--n_steps', type=int, default=20, help='number of time steps to simulate for')
  parser.add_argument('--dt', type=float, default=1./60., help='simulation step size')
  parser.add_argument('--ball_pos', type=int, nargs=2, default=None, help='ball initial position (randomly generated if not supplied)')
  parser.add_argument('--ball_vel', type=float, nargs=2, default=None, help='ball initial velocity (randomly generated if not supplied)')
  parser.add_argument('--skip_size', type=int, default=1, help='number of frames to skip in dumped trajectories')
  return parser.parse_args()


def build_wall(start, end, space):
  wall = pymunk.Segment(space.static_body, start, end, 0.0)
  wall.elasticity = ELASTICITY
  wall.friction = FRICTION
  return wall


def build_ball(init_pos, init_vel):
  inertia = pymunk.moment_for_circle(BALL_M, 0, BALL_R, (0, 0))
  body = pymunk.Body(BALL_M, inertia)
  body.position = init_pos
  body.velocity = init_vel
  ball = pymunk.Circle(body, BALL_R, (0, 0))
  ball.elasticity = ELASTICITY
  ball.friction = FRICTION
  return ball


def build_space():
  space = pymunk.Space()
  space.gravity = GRAVITY
  walls = [build_wall((0, 0), (0, SCREEN_SIZE), space),
           build_wall((0, 0), (SCREEN_SIZE, 0), space),
           build_wall((SCREEN_SIZE, 0), (SCREEN_SIZE, SCREEN_SIZE), space),
           build_wall((0, SCREEN_SIZE), (SCREEN_SIZE, SCREEN_SIZE), space)]
  space.add(walls)
  return space



def generate_trajectory(space, n_steps):
  x_offset = random.randint(-100, 100)
  y_offset = random.randint(-100, 100)
  ball_init_pos = (SCREEN_SIZE//2 + x_offset, SCREEN_SIZE//2 + y_offset)
  ball_init_vel = (random.randint(-500, 500), random.randint(-500, 500))
  if args.ball_pos:
    ball_init_pos = tuple(args.ball_pos)
  if args.ball_vel:
    ball_init_vel = tuple(args.ball_vel)
  ball = build_ball(ball_init_pos, ball_init_vel)
  space.add(ball, ball.body)

  trajectory = []
  for _ in range(n_steps):
    space.step(args.dt)
    trajectory.append(ball.body.position)

  space.remove(ball, ball.body)
  return np.array(trajectory) / float(SCREEN_SIZE)


if __name__=='__main__':
  args = cli_args()
  if args.seed:
    random.seed(args.seed)

  space = build_space()

  bar = progressbar.ProgressBar()
  trajectories = []
  for _ in bar(range(args.n_trajectories)):
    trajectories.append(generate_trajectory(space, args.skip_size*args.n_steps)[::args.skip_size])
  fname = '{}trajectories_{}steps_{}skip_{}seed.pkl'.format(args.n_trajectories, args.n_steps, args.skip_size, args.seed if args.seed else 'no')
  with open(fname, 'wb') as f:
    pickle.dump(trajectories, f, protocol=2)

