#!/usr/bin/env python
"""
Rollout script with per-action video recording.

Each pick-and-place action generates a continuous video clip capturing the full
robot motion. A keyframe index marks the moment "place contact" is detected.
Rendering runs in a background thread so robot motion speed is unaffected.

Output structure:
    recordings/{task}/ep_{seed:06d}/
        cameras.json
        metadata.json
        action_{t:03d}/
            cam{i}_rgb.mp4        # continuous motion video, 16fps, 832x480
            cam{i}_depth.npy      # [T, 480, 640] float32
            cam{i}_segm.npy       # [T, 480, 640] int32
            keyframe.json         # {"place_contact_frame": int, "total_frames": int}

Usage:
    python rollout_record.py --gpu=0 --task=cable-shape --hz=240 --disp --num_demos=5
"""
import os
import sys
import time
import json
import argparse

import cv2
import numpy as np
import pybullet as p
import tensorflow as tf

from ravens import tasks, cameras
from ravens.environment import Environment

CAM_NAMES = ['front', 'left', 'right']
RECORD_FPS = 16
VIDEO_SIZE = (832, 480)

_FRAME_INTERVAL = 1.0 / RECORD_FPS  # seconds between captured frames


def build_camera_json(cam_configs):
    """Serialize camera configs to JSON-friendly list."""
    result = []
    for i, cfg in enumerate(cam_configs):
        intr = cfg['intrinsics']
        K = [[intr[0], intr[1], intr[2]],
             [intr[3], intr[4], intr[5]],
             [intr[6], intr[7], intr[8]]]
        rot_mat = np.array(p.getMatrixFromQuaternion(cfg['rotation'])).reshape(3, 3).tolist()
        result.append({
            'cam_id': i,
            'name': CAM_NAMES[i],
            'image_size': list(cfg['image_size']),
            'K': K,
            'position': list(cfg['position']),
            'rotation_quat': list(cfg['rotation']),
            'rotation_matrix': rot_mat,
            'zrange': list(cfg['zrange']),
        })
    return result


class RecordingEnvironment(Environment):
    """Captures frames inside movej() with frame-skipping to minimize overhead."""

    def __init__(self, disp=False, hz=240):
        super().__init__(disp, hz)
        self._cam_configs = None
        self._frame_bufs = None
        self._depth_bufs = None
        self._segm_bufs  = None
        self._place_contact_frame = -1
        self._recording = False
        self._last_frame_time = 0.0

    def start_recording(self, cam_configs):
        n = len(cam_configs)
        self._cam_configs = cam_configs
        self._frame_bufs = [[] for _ in range(n)]
        self._depth_bufs = [[] for _ in range(n)]
        self._segm_bufs  = [[] for _ in range(n)]
        self._place_contact_frame = -1
        self._last_frame_time = 0.0
        self._recording = True

    def stop_recording(self):
        """Stop recording; render cam1/cam2 once at the boundary frame."""
        self._recording = False
        if self._cam_configs and self._frame_bufs and self._frame_bufs[0]:
            # Render side cameras at the action boundary (last frame of this clip)
            for i in range(1, len(self._cam_configs)):
                color, depth, segm = self.render(self._cam_configs[i])
                frame = cv2.resize(color[:, :, ::-1], VIDEO_SIZE)
                self._frame_bufs[i].append(frame)
                self._depth_bufs[i].append(depth.astype(np.float32))
                self._segm_bufs[i].append(segm.astype(np.int32))
            # Keyframe = last frame = boundary between this action and the next
            self._place_contact_frame = len(self._frame_bufs[0]) - 1
        return self._frame_bufs, self._depth_bufs, self._segm_bufs, self._place_contact_frame

    def _grab_frame(self):
        """Render front camera at fixed time intervals during motion."""
        if not self._recording or self._cam_configs is None:
            return
        now = time.time()
        if now - self._last_frame_time < _FRAME_INTERVAL:
            return
        self._last_frame_time = now
        color, depth, segm = self.render(self._cam_configs[0])
        frame = cv2.resize(color[:, :, ::-1], VIDEO_SIZE)
        self._frame_bufs[0].append(frame)
        self._depth_bufs[0].append(depth.astype(np.float32))
        self._segm_bufs[0].append(segm.astype(np.int32))

    def movej(self, targj, speed=0.01, t_lim=20):
        t0 = time.time()
        while (time.time() - t0) < t_lim:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                self._grab_frame()
                return True
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self._grab_frame()
            time.sleep(0.001)
        print(f'Warning: movej exceeded {t_lim} sec timeout. Skipping.')
        return False

    def pick_place(self, pose0, pose1):
        """Override to mark place-contact keyframe."""
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        if hasattr(self.task, 'primitive_params'):
            ts = self.task.task_stage
            if 'prepick_z' in self.task.primitive_params[ts]:
                prepick_z = self.task.primitive_params[ts]['prepick_z']
            speed       = self.task.primitive_params[ts]['speed']
            delta_z     = self.task.primitive_params[ts]['delta_z']
            postpick_z  = self.task.primitive_params[ts]['postpick_z']
            preplace_z  = self.task.primitive_params[ts]['preplace_z']
            pause_place = self.task.primitive_params[ts]['pause_place']

        def_IDs = []
        if hasattr(self.task, 'def_IDs'):
            def_IDs = self.task.def_IDs

        success = True
        pick_position = np.array(pose0[0])
        pick_rotation = np.array(pose0[1])
        prepick_position = pick_position.copy()
        prepick_position[2] = prepick_z

        # --- PICK PHASE ---
        prepick_pose = np.hstack((prepick_position, pick_rotation))
        success &= self.movep(prepick_pose)
        target_pose = prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

        while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
            target_pose += delta
            success &= self.movep(target_pose)

        self.ee.activate(self.objects, def_IDs)

        if self.is_softbody_env() or self.is_new_cable_env():
            prepick_pose[2] = postpick_z
            success &= self.movep(prepick_pose, speed=speed)
            time.sleep(pause_place)
        elif isinstance(self.task, tasks.names['cable']):
            prepick_pose[2] = 0.03
            success &= self.movep(prepick_pose, speed=0.001)
        else:
            prepick_pose[2] += pick_position[2]
            success &= self.movep(prepick_pose)
        pick_success = self.ee.check_grasp()

        if pick_success:
            place_position = np.array(pose1[0])
            place_rotation = np.array(pose1[1])
            preplace_position = place_position.copy()
            preplace_position[2] = 0.3 + pick_position[2]

            # --- PLACE PHASE ---
            preplace_pose = np.hstack((preplace_position, place_rotation))
            if self.is_softbody_env() or self.is_new_cable_env():
                preplace_pose[2] = preplace_z
                success &= self.movep(preplace_pose, speed=speed)
                time.sleep(pause_place)
            elif isinstance(self.task, tasks.names['cable']):
                preplace_pose[2] = 0.03
                success &= self.movep(preplace_pose, speed=0.001)
            else:
                success &= self.movep(preplace_pose)

            target_pose = preplace_pose.copy()
            while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
                target_pose += delta
                success &= self.movep(target_pose)

            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            preplace_pose[2] = final_z
            success &= self.movep(preplace_pose)
        else:
            self.ee.release()
            prepick_pose[2] = final_z
            success &= self.movep(prepick_pose)
        return success


def save_action_recording(ep_dir, action_idx, frame_bufs, depth_bufs, segm_bufs, place_frame):
    """Save one action's recording to disk."""
    action_dir = os.path.join(ep_dir, f'action_{action_idx:03d}')
    os.makedirs(action_dir, exist_ok=True)

    n_cams = len(frame_bufs)
    for i in range(n_cams):
        if not frame_bufs[i]:
            continue
        # Write video
        path = os.path.join(action_dir, f'cam{i}_rgb.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, RECORD_FPS, VIDEO_SIZE)
        for frame in frame_bufs[i]:
            writer.write(frame)
        writer.release()

        # Write depth & segm
        np.save(os.path.join(action_dir, f'cam{i}_depth.npy'), np.stack(depth_bufs[i]))
        np.save(os.path.join(action_dir, f'cam{i}_segm.npy'),  np.stack(segm_bufs[i]))

    total_frames = len(frame_bufs[0]) if frame_bufs[0] else 0
    keyframe_info = {
        'place_contact_frame': place_frame,
        'total_frames': total_frames,
    }
    with open(os.path.join(action_dir, 'keyframe.json'), 'w') as f:
        json.dump(keyframe_info, f, indent=2)


def rollout_and_record(agent, env, task, cam_configs, ep_dir):
    """Run one episode, record each action as a separate video clip.

    Returns:
        total_reward: float
        num_actions: int
    """
    obs = env.reset(task)
    info = env.info
    total_reward = 0
    action_idx = 0

    for t in range(task.max_steps):
        act = agent.act(obs, info)
        if not act or not act.get('primitive'):
            obs, reward, done, info = env.step(act)
            total_reward += reward
            if done:
                break
            continue

        # Start recording before executing the action
        env.start_recording(cam_configs)
        obs, reward, done, info = env.step(act)
        frame_bufs, depth_bufs, segm_bufs, place_frame = env.stop_recording()
        total_reward += reward

        # Save this action's clip
        save_action_recording(ep_dir, action_idx, frame_bufs, depth_bufs, segm_bufs, place_frame)
        n_frames = len(frame_bufs[0]) if frame_bufs[0] else 0
        print(f'    action {action_idx}: {n_frames} frames, keyframe={place_frame}')
        action_idx += 1

        if done:
            break

    return total_reward, action_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',       default='0')
    parser.add_argument('--disp',      action='store_true')
    parser.add_argument('--task',      default='cable-shape')
    parser.add_argument('--hz',        default=240.0, type=float)
    parser.add_argument('--num_demos', default=5,     type=int)
    parser.add_argument('--save_dir',  default='recordings')
    parser.add_argument('--gpu_mem_limit', default=None)
    args = parser.parse_args()

    # Configure GPU.
    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices('GPU')
    if not gpus:
        print('No GPUs detected. Running with CPU.')
    else:
        cfg.set_visible_devices(gpus[int(args.gpu)], 'GPU')
    if args.gpu_mem_limit is not None:
        dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=1024 * int(args.gpu_mem_limit))]
        cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    task = tasks.names[args.task]()
    task.mode = 'train'
    cam_configs = cameras.RealSenseD415.CONFIG
    cam_json = build_camera_json(cam_configs)

    env = RecordingEnvironment(args.disp, hz=args.hz)

    seed_offset = 0
    ep_idx = 0
    while ep_idx < args.num_demos:
        seed = ep_idx + seed_offset
        np.random.seed(seed)
        print(f'\nEpisode {ep_idx + 1}/{args.num_demos}  seed={seed}')

        ep_dir = os.path.join(args.save_dir, args.task, f'ep_{seed:06d}')
        os.makedirs(ep_dir, exist_ok=True)

        with open(os.path.join(ep_dir, 'cameras.json'), 'w') as f:
            json.dump(cam_json, f, indent=2)

        agent = task.oracle(env)
        reward, num_actions = rollout_and_record(agent, env, task, cam_configs, ep_dir)

        if num_actions == 0:
            seed_offset += 1
            print(f'  Skipping (no actions), re-sampling seed={seed + 1}')
            continue

        meta = {
            'seed':        seed,
            'reward':      float(reward),
            'num_actions': num_actions,
            'success':     bool(reward > 0.99),
            'task':        args.task,
        }
        with open(os.path.join(ep_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'  reward={reward:.4f}  actions={num_actions}  success={meta["success"]}')
        ep_idx += 1

    env.stop()
    del env
    print(f'\nDone. Recordings saved to {args.save_dir}/{args.task}/')
