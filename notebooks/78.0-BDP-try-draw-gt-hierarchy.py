# import graph_tool as gt
from graph_tool.collection import data
from graph_tool import inference
from graph_tool import draw
from graph_tool.draw import draw_hierarchy

g = data["celegansneural"]
state = inference.minimize_nested_blockmodel_dl(g, deg_corr=True)
draw_hierarchy(state, output="celegansneural_nested_mdl.pdf")
