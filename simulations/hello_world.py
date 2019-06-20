#%%
from sacred import Experiment
from sacred.observers import SlackObserver, FileStorageObserver
from sacred import SETTINGS
from src.utils import save_obj

SETTINGS

ex = Experiment("config_demo")

slack_obs = SlackObserver.from_config("slack.json")
ex.observers.append(slack_obs)
fso = FileStorageObserver.create("./simulations/runs/hello_world")
ex.observers.append(fso)
print(dir(fso))


@ex.config
def my_config1():
    a = 10  # noqa: F841
    b = "test"  # noqa: F841


@ex.capture
def print_a_and_b(a, b):
    print("a =", a)
    print("b =", b)


@ex.automain
def my_main(a, b):
    print(a)
    print(b)
    print(fso.run_entry)
    print(fso.info)
    # print(fso._id)
    print(fso.dir)
    save_obj(b, fso, "test")
    return 1

