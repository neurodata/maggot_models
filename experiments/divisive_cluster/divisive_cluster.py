
# %% [markdown]
# ## Clustering

from graspologic.cluster import DivisiveCluster

# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split

params = [
    {"d": 8, "bic_ratio": 0, "min_split": 32},
    {"d": 8, "bic_ratio": 0.95, "min_split": 32},
]

for p in params:
    print(p)
    d = p["d"]
    bic_ratio = p["bic_ratio"]
    min_split = p["min_split"]
    X = embedding[:, :d]
    basename = f"-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"

    currtime = time.time()
    np.random.seed(8888)
    mc = BinaryCluster(
        "0",
        n_init=50,  # number of initializations for GMM at each stage
        meta=nodes,  # stored for plotting and adding labels
        X=X,  # input data that actually matters
        bic_ratio=bic_ratio,
        reembed=False,
        min_split=min_split,
    )

    mc.fit(n_levels=n_levels, metric=metric)
    print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for clustering")

    cluster_meta = mc.meta

    # save results
    cluster_meta.to_csv("meta" + basename)

    print()

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")