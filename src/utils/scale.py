
import pickle

def scale_memory(context, query, scale=3):
    with open(query, "rb") as f:
        query_metas = pickle.load(f)

    labels_contents = [ref for q_meta  in query_metas for ref in q_meta['reference_list']]
    copy_reference = [f"{ref}_copy{bias}" for bias, ref in enumerate(context * scale) if ref not in labels_contents]
    context.extend(copy_reference) 