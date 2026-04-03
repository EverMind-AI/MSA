import os
import sys
sys.path.append(os.getcwd())
from src.msa.model import MSAForCausalLM
from src.msa.configuration_msa import MSAConfig
from src.utils.common import print_model_stats
from transformers import AutoTokenizer

def save_checkpoint(model, tokenizer, save_model_path):
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)

def main(origin_model_path, save_model_path):
    router_layer_idx = os.environ.get("ROUTER_LAYER_IDX", "all")
    aux_loss = os.environ.get("AUX_LOSS", "false") == "true"
    lmloss_weigth = float(os.environ.get("LMLOSS_WEIGHT", 1.0))
    auxloss_weight = float(os.environ.get("AUX_LOSS_WEIGHT", 0.1))
    recloss_weight = float(os.environ.get("REC_LOSS_WEIGHT", 0.0))
    ansloss_weight = float(os.environ.get("ANS_LOSS_WEIGHT", 1.0))
    aux_loss_method = os.environ.get("AUX_LOSS_METHOD", "INFONCE")  # INFONCE, BCE, INFONCE_DECOUPLE, INFONCE_DECOUPLE_FOCAL 
    decouple_router = os.environ.get("DECOUPLE_ROUTER", "false").lower() == "true"
    rewrite_position = os.environ.get("REWRITE_POSITION", "false") == "true"

    top_k_docs = int(os.environ.get("TOP_K_DOCS", 2))
    pooling_kernel_size = int(os.environ.get("POOLING_KERNEL_SIZE", 2))

    head_reduce_method = os.environ.get("HEAD_REDUCE_METHOD", "max")
    query_reduce_method = os.environ.get("QUERY_REDUCE_METHOD", "max")
    chunk_reduce_method = os.environ.get("CHUNK_REDUCE_METHOD", "max")
    decouple_pooling_mode = os.environ.get("DECOUPLE_POOLING_MODE", "mean")
    infonce_loss_temp = float(os.environ.get("INFONCE_LOSS_TEMP", 0.1))

    msa_config = {
        "router_layer_idx": router_layer_idx,
        "aux_loss": aux_loss,
        "lmloss_weigth": lmloss_weigth,
        "auxloss_weight": auxloss_weight,
        "recloss_weight": recloss_weight,
        "ansloss_weight": ansloss_weight,
        "aux_loss_method": aux_loss_method,
        "decouple_router": decouple_router,
        "rewrite_position": rewrite_position,
        "top_k_docs": top_k_docs,
        "pooling_kernel_size": pooling_kernel_size,
        "infonce_loss_temp": infonce_loss_temp,
        "head_reduce_method": head_reduce_method,
        "query_reduce_method": query_reduce_method,
        "chunk_reduce_method": chunk_reduce_method,
        "decouple_pooling_mode": decouple_pooling_mode,
    }
    # 使用 MSAConfig，它会自动将 msa_config 转换为 DotDict
    config = MSAConfig.from_pretrained(origin_model_path)
    config.msa_config = msa_config  # MSAConfig 会自动转换为 DotDict
    tokenizer = AutoTokenizer.from_pretrained(origin_model_path)
    model = MSAForCausalLM.from_pretrained(
        origin_model_path,
        config=config,
        torch_dtype="bfloat16",
    )
    print_model_stats(model)

    # save
    save_checkpoint(model, tokenizer, save_model_path)

if __name__ == "__main__":
    origin_model_path = sys.argv[1]
    save_model_path = sys.argv[2]
    main(origin_model_path, save_model_path)
    print(f"Model has been saved to : {save_model_path}")
    print("Done")