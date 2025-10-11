"""
Module to register Qwen3-Omni model architectures with both transformers and AWQ libraries.
This solves the issue where transformers and AWQ don't recognize the 'qwen3_omni_moe' model type.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, Tuple, Union
import warnings


class Qwen3OmniMoeConfig(PretrainedConfig):
    """
    Configuration class for Qwen3OmniMoe model.
    """
    model_type = "qwen3_omni_moe"
    is_composition = False
    
    def __init__(
        self,
        vocab_size=152064,
        hidden_size=2048,
        intermediate_size=768,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        mlp_only_layers=[],
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_act="silu",
        max_position_embeddings=65536,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        bos_token_id=151644,
        eos_token_id=151645,
        pad_token_id=151643,
        tie_word_embeddings=False,
        rope_theta=1000000,
        attention_dropout=0.0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.mlp_only_layers = mlp_only_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        
        # Thinker configuration - these are critical for the multimodal model
        self.thinker_config = kwargs.pop('thinker_config', {})
        self.assistant_token_id = kwargs.pop('assistant_token_id', 77091)
        self.enable_audio_output = kwargs.pop('enable_audio_output', False)
        self.im_end_token_id = kwargs.pop('im_end_token_id', 151645)
        self.im_start_token_id = kwargs.pop('im_start_token_id', 151644)
        self.system_token_id = kwargs.pop('system_token_id', 8948)
        self.user_token_id = kwargs.pop('user_token_id', 872)
        
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen3OmniMoeForConditionalGeneration(PreTrainedModel):
    """
    Placeholder implementation of Qwen3OmniMoe model.
    This is a minimal implementation to allow loading of the model configuration.
    """
    config_class = Qwen3OmniMoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3OmniMoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    supports_gradient_checkpointing = True

    def __init__(self, config, fuse_layers=False, use_exllama=False, **kwargs):
        super().__init__(config)
        # For AWQ compatibility, we need to have the basic structure that AWQ expects
        # This is a minimal implementation to allow loading with AWQ
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList([])
        self.model.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Handle the parameters (we can ignore them for our stub implementation)
        self.fuse_layers = fuse_layers
        self.use_exllama = use_exllama
        
        # Handle any other unexpected keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize weights
        self.post_init()

    def post_init(self):
        """
        A method executed at the end of each model's initialization, to execute
        arbitrary code after the model has been initialized.
        """
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        # Initialize weights using the config's initializer_range
        with torch.no_grad():
            # Initialize lm_head
            if hasattr(self.lm_head, 'weight') and self.lm_head.weight is not None:
                self.lm_head.weight.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, *args, **kwargs):
        """
        Forward method placeholder.
        This should be replaced with the actual forward implementation.
        """
        raise NotImplementedError("Qwen3OmniMoeForConditionalGeneration is a placeholder implementation. "
                                  "For actual usage, you need the complete model implementation.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Override from_pretrained to handle AWQ models properly
        """
        # If this is an AWQ quantized model, we need to handle it specially
        # Let users know that this is a stub implementation
        print(f"Loading Qwen3OmniMoe model from {pretrained_model_name_or_path}")
        print("Note: This is a stub implementation. For full functionality, use the official model implementation.")
        
        # Call the parent method to handle the loading
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        """
        Override save_pretrained to maintain compatibility
        """
        return super().save_pretrained(*args, **kwargs)


def register_qwen3_omni_models():
    """
    Register Qwen3OmniMoe models with the transformers library.
    This function should be called before attempting to load the model.
    """
    # Register the configuration
    if "qwen3_omni_moe" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("qwen3_omni_moe", Qwen3OmniMoeConfig)
        print("Qwen3OmniMoeConfig registered with transformers CONFIG_MAPPING")
    else:
        print("Qwen3OmniMoeConfig already registered")
    
    # Register the model for causal LM
    if Qwen3OmniMoeConfig not in MODEL_FOR_CAUSAL_LM_MAPPING:
        MODEL_FOR_CAUSAL_LM_MAPPING.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)
        print("Qwen3OmniMoeForConditionalGeneration registered with MODEL_FOR_CAUSAL_LM_MAPPING")
    else:
        print("Qwen3OmniMoeForConditionalGeneration already registered")

    # Additional registration for Auto classes
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        AutoConfig.register("qwen3_omni_moe", Qwen3OmniMoeConfig)
        AutoModel.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)
        AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)
        print("Qwen3OmniMoe models registered with Auto classes")
    except ImportError:
        print("Could not register with Auto classes, but basic registration completed")


def register_qwen3_omni_with_awq():
    """
    Register Qwen3OmniMoe with the AWQ library.
    This is needed because AWQ has its own model type checking.
    """
    print("Attempting to register qwen3_omni_moe with AWQ...")
    
    # Direct approach: Patch the exact location where the error occurs
    # The error is: KeyError: 'qwen3_omni_moe' in TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
    try:
        import awq.models.base
        # Ensure the dictionary exists
        if not hasattr(awq.models.base, 'TRANSFORMERS_AUTO_MAPPING_DICT'):
            awq.models.base.TRANSFORMERS_AUTO_MAPPING_DICT = {}
        
        # Add our model type mapping
        awq.models.base.TRANSFORMERS_AUTO_MAPPING_DICT["qwen3_omni_moe"] = "Qwen3OmniMoeForConditionalGeneration"
        print("qwen3_omni_moe registered with AWQ's TRANSFORMERS_AUTO_MAPPING_DICT")
    except Exception as e:
        print(f"Failed to patch TRANSFORMERS_AUTO_MAPPING_DICT: {e}")
    
    # Also patch the model map
    try:
        import awq.models.auto
        # Ensure the dictionary exists
        if not hasattr(awq.models.auto, 'AWQ_CAUSAL_LM_MODEL_MAP'):
            awq.models.auto.AWQ_CAUSAL_LM_MODEL_MAP = {}
        
        # Create a minimal AWQ wrapper that delegates to our registered model
        class Qwen3OmniMoeAWQWrapper:
            @classmethod
            def from_quantized(cls, *args, **kwargs):
                # Extract the quant_path from args (first positional argument)
                quant_path = args[0] if args else kwargs.get('quant_path', '')
                print(f"Loading qwen3_omni_moe model via AWQ wrapper: {quant_path}")
                
                # Load using our registered model with proper parameters
                model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    quant_path, **kwargs)
                
                # Return a mock AWQ structure that mimics the expected interface
                result = type('MockAWQResult', (), {})()
                result.model = model
                # Add a mock quant_config attribute
                result.model.quant_config = type('MockQuantConfig', (), {
                    'zero_point': True,
                    'q_group_size': 128,
                    'w_bit': 4,
                    'version': 'Qwen3OmniMoe'
                })()
                return result
        
        awq.models.auto.AWQ_CAUSAL_LM_MODEL_MAP["qwen3_omni_moe"] = Qwen3OmniMoeAWQWrapper
        print("qwen3_omni_moe registered with AWQ_CAUSAL_LM_MODEL_MAP")
    except Exception as e:
        print(f"Failed to patch AWQ_CAUSAL_LM_MODEL_MAP: {e}")
        
        # Final fallback: Monkey-patch the specific error location
        try:
            import awq.models.base
            original_from_quantized = getattr(awq.models.base.BaseAWQForCausalLM, 'from_quantized', None)
            
            @classmethod
            def patched_from_quantized(cls, *args, **kwargs):
                # Extract the quant_path from args (first positional argument)
                quant_path = args[0] if args else kwargs.get('quant_path', '')
                
                from transformers import AutoConfig
                # Extract trust_remote_code from kwargs to avoid duplicates
                trust_remote_code = kwargs.get('trust_remote_code', True)
                config = AutoConfig.from_pretrained(quant_path, trust_remote_code=trust_remote_code)
                
                if config.model_type == "qwen3_omni_moe":
                    print("Handling qwen3_omni_moe model with direct patch...")
                    # Load using our registered model
                    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                        quant_path, **kwargs)
                    
                    # Return in a format that mimics AWQ's expectation
                    result = type('MockResult', (), {})()
                    result.model = model
                    result.model.quant_config = type('MockQuantConfig', (), {
                        'zero_point': True,
                        'q_group_size': 128,
                        'w_bit': 4,
                        'version': 'Qwen3OmniMoe'
                    })()
                    return result
                else:
                    # Call original if available
                    if original_from_quantized:
                        return original_from_quantized(cls, *args, **kwargs)
                    else:
                        raise NotImplementedError("Original AWQ from_quantized not available")
            
            awq.models.base.BaseAWQForCausalLM.from_quantized = patched_from_quantized
            print("Direct patch applied to BaseAWQForCausalLM.from_quantized")
        except Exception as patch_error:
            print(f"Failed to apply direct patch: {patch_error}")


def register_all():
    """
    Register the Qwen3OmniMoe model with both transformers and AWQ libraries.
    """
    register_qwen3_omni_models()
    register_qwen3_omni_with_awq()


# Run registration when module is imported
if __name__ != "__main__":
    register_all()