#%%
import gzip

edgelist_loc = "maggot_models/data/processed/2020-09-23/G.edgelist"
with open(edgelist_loc, "rb") as f:
    raw_edgelist = f.read()

raw_kb = len(raw_edgelist) / 1000

gzip_edgelist = gzip.compress(raw_edgelist)
gzip_kb = len(gzip_edgelist) / 1000

print(f"Raw edgelist size: {raw_kb} kb")
print(f"Gzipped edgelist size: {gzip_kb} kb")
print(f"Compression ratio: {gzip_kb/raw_kb} kb")