import numpy as np
import rainy
from rainy import agents, envs, net
from acvp.datagen import ImageWriterHook


def test_image_hook_atari() -> None:
    c = rainy.Config()
    hook = ImageWriterHook(out_dir="/tmp/rainy-acvp/imagehook-test")
    c.eval_hooks.append(ImageWriterHook(out_dir="/tmp/rainy-acvp/imagehook-test"))
    c.set_net_fn("dqn", net.value.dqn_conv())
    c.set_env(lambda: envs.Atari("Breakout"))
    c.eval_env = envs.Atari("Breakout")
    ag = agents.DQNAgent(c)
    c.initialize_hooks()
    _ = ag.eval_episode()
    ag.close()
    images = np.load(hook.out_dir.joinpath("ep1.npz"))
    assert images["states"][0].shape == (210, 160, 3)
    assert len(images["actions"].shape) == 1
