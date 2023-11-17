import os

import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import animation

def save_replay_buffer(replay_buffer, filename):
    # Open a file and use dump()
    #with open('/root/applr_ws/src/APPLR_social_nav/training/pic4rl/results/rb_TD3_100k_pr_ns4.npz', 'wb') as file:
    file = "/root/gym_ws/src/PIC4rl_gym/training/pic4rl/results/rb_TD3_100k_pr_ns4.npz"
    # A new file will be created
    #pickle.dump(replay_buffer, file)
    replay_buffer.save_transitions(file, safe=True)

def load_replay_buffer(rb_path):
    # Open the file in binary mode
    with open(rb_path, 'rb') as file:
      
        # Call load method to deserialze
        replay_buffer = pickle.load(file)
      
        return replay_buffer
        
def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def restore_latest_n_traj(dirname, n_path=10, max_steps=None):
    assert os.path.isdir(dirname)
    filenames = get_filenames(dirname, n_path)
    return load_trajectories(filenames, max_steps)


def get_filenames(dirname, n_path=None):
    import re
    itr_reg = re.compile(
        r"step_(?P<step>[0-9]+)_epi_(?P<episodes>[0-9]+)_return_(-?)(?P<return_u>[0-9]+).(?P<return_l>[0-9]+).pkl")

    itr_files = []
    for _, filename in enumerate(os.listdir(dirname)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('step')
            itr_files.append((itr_count, filename))

    n_path = n_path if n_path is not None else len(itr_files)
    itr_files = sorted(itr_files, key=lambda x: int(
        x[0]), reverse=True)[:n_path]
    filenames = []
    for itr_file_and_count in itr_files:
        filenames.append(os.path.join(dirname, itr_file_and_count[1]))
    return filenames


def load_trajectories(filenames, max_steps=None):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))

    def get_obs_and_act(path):
        obses = path['obs']
        next_obses = path['next_obs']
        actions = path['act']
        if max_steps is not None:
            return obses[:max_steps], next_obses[:max_steps], actions[:max_steps]
        else:
            return obses, next_obses, actions

    for i, path in enumerate(paths):
        if i == 0:
            obses, next_obses, acts = get_obs_and_act(path)
        else:
            obs, next_obs, act = get_obs_and_act(path)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            acts = np.vstack((act, acts))
    return {'obses': obses, 'next_obses': next_obses, 'acts': acts}


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    """
    Convert frames to gif file
    """
    assert len(frames) > 0
    plt.figure(figsize=(frames[0].shape[1] / 72.,
                        frames[0].shape[0] / 72.), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    # TODO: interval should be 1000 / fps ?
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=interval)
    output_path = "{}/{}.gif".format(save_dir, prefix)
    anim.save(output_path, writer='imagemagick', fps=fps)
