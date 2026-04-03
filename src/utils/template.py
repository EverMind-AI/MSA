QWEN3_TEMPLATE={
    "prompt": "<|im_start|>user\n{prompt}<|im_end|>\n",
    "response": "<|im_start|>assistant\n<think>\n{think_content}\n</think>\n\n{output}<|im_end|>"
}

QWEN3_INSTRUCT_TEMPLATE={
    "prompt": "<|im_start|>user\n{prompt}<|im_end|>\n",
    "response": "<|im_start|>assistant\n{output}<|im_end|>"
}