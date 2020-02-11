# TODO: now write simple function which shows all of the plots I want in general when
#    doing a clustering (modularity or cell type)


# def augment_classes(class_labels, lineage_labels, fill_unk=True):
#     if fill_unk:
#         classlin_labels = class_labels.copy()
#         fill_inds = np.where(class_labels == "unk")[0]
#         classlin_labels[fill_inds] = lineage_labels[fill_inds]
#         used_inds = np.array(list(CLASS_IND_DICT.values()))
#         unused_inds = np.setdiff1d(range(len(cc.glasbey_light)), used_inds)
#         lineage_color_dict = dict(
#             zip(np.unique(lineage_labels), np.array(cc.glasbey_light)[unused_inds])
#         )
#         color_dict = {**CLASS_COLOR_DICT, **lineage_color_dict}
#         hatch_dict = {}
#         for key, val in color_dict.items():
#             if key[0] == "~":
#                 hatch_dict[key] = "//"
#             else:
#                 hatch_dict[key] = ""
#     else:
#         color_dict = "class"
#         hatch_dict = None
#     return classlin_labels, color_dict, hatch_dict


# from .visualize import barplot_text


# def plot_cluster_results(
#     partition, class_labels, lineage_labels, order=None, title=None, savename=None
# ):
#     """Convenience function for generating many plots and saving them after a clustering
#     """

#     if order == "similarity":
#         pass
#     elif order == "sensorimotor":
#         pass
#     elif order == "signal-flow":
#         pass
#     else:
#         raise ValueError("order is not a recognized key")

#     classlin_labels, color_dict, hatch_dict = augment_classes(
#         class_labels, lineage_labels
#     )

#     # Barplots
#     _, _, order = barplot_text(
#         partition,
#         classlin_labels,
#         norm_bar_width=True,
#         color_dict=color_dict,
#         hatch_dict=hatch_dict,
#         title=title,
#         figsize=(24, 18),
#         return_order=True,
#     )
#     stashfig(savename + "barplot-mergeclasslin-props")
#     category_order = np.unique(block_label)[order]
