#%%
from sacred import Experiment
from sacred.observers import SlackObserver, FileStorageObserver
from sacred import SETTINGS

SETTINGS

ex = Experiment("config_demo")

# slack_obs = SlackObserver.from_config("slack.json")
# ex.observers.append(slack_obs)
ex.observers.append(FileStorageObserver.create("my_runs"))


@ex.config
def my_config1():
    a = 10
    b = "test"


@ex.capture
def print_a_and_b(a, b):
    print("a =", a)
    print("b =", b)


@ex.automain
def my_main(a, b):
    print(a)
    print(b)
    return 1

