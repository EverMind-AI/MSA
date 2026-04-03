"""MSA (Memory Sparse Attention) Configuration"""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class DotDict(dict):
    """支持点号访问的字典类"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getstate__(self):
        return dict(self)
    
    def __setstate__(self, state):
        self.update(state)


class MSAConfig(Qwen3Config):
    """
    MSA 模型的配置类，继承自 Qwen3Config。
    
    主要功能：确保 msa_config 在加载时自动转换为 DotDict，
    支持使用点号访问属性（如 config.msa_config.pad_free）
    """
    
    model_type = "msa"
    
    def __init__(self, msa_config=None, **kwargs):
        super().__init__(**kwargs)
        if msa_config is not None:
            self.msa_config = DotDict(msa_config) if not isinstance(msa_config, DotDict) else msa_config
    
    def __setattr__(self, name, value):
        """重写 __setattr__，确保设置 msa_config 时自动转换为 DotDict"""
        if name == "msa_config" and isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super().__setattr__(name, value)
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        从字典创建配置对象时，确保 msa_config 被转换为 DotDict。
        这是关键方法，AutoConfig.from_pretrained() 最终会调用这个方法。
        """
        # 先调用父类的 from_dict
        config = super().from_dict(config_dict, **kwargs)
        
        # 确保 msa_config 是 DotDict
        if hasattr(config, 'msa_config') and isinstance(config.msa_config, dict) and not isinstance(config.msa_config, DotDict):
            config.msa_config = DotDict(config.msa_config)
        
        return config
