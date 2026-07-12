# model_loader.py
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import BASE_MODEL, ADAPTERS, DEVICE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.current_adapter = None
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def load_base_model(self):
        """Load base model once — reuse for all languages"""
        if self.base_model is not None:
            logger.info("Base model already loaded")
            return

        logger.info(f"Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=self.bnb_config,
            device_map={"": 0},
        )
        self.base_model.eval()
        logger.info("✅ Base model loaded")

    
    def load_adapter(self, lang: str):
        """Hot-swap LoRA adapter for given language"""
        if lang not in ADAPTERS:
            raise ValueError(f"Unknown language: {lang}")

        if self.base_model is None:
            self.load_base_model()

        if self.current_adapter == lang:
            logger.info(f"Adapter {lang} already loaded")
            return

        logger.info(f"Loading {lang} adapter...")
        self._patch_adapter_config(lang)

        if not isinstance(self.base_model, PeftModel):
            # First adapter: wrap the base model, name it explicitly
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                ADAPTERS[lang],
                adapter_name=lang,
                is_trainable=False,
            )
        elif lang not in self.base_model.peft_config:
            # Already a PeftModel, but this adapter isn't attached yet
            self.base_model.load_adapter(
                ADAPTERS[lang], adapter_name=lang, is_trainable=False
            )

        self.base_model.set_adapter(lang)
        self.base_model.eval()
        self.current_adapter = lang
        logger.info(f"✅ {lang} adapter loaded")

    def _patch_adapter_config(self, lang: str):
        """Remove unknown keys from adapter_config.json"""
        import json, os

        config_path = os.path.join(ADAPTERS[lang], "adapter_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        valid_keys = {
            "peft_type", "auto_mapping", "base_model_name_or_path",
            "revision", "task_type", "inference_mode",
            "r", "target_modules", "lora_alpha", "lora_dropout",
            "fan_in_fan_out", "bias", "modules_to_save",
            "init_lora_weights", "layers_to_transform", "layers_pattern",
            "rank_pattern", "alpha_pattern", "megatron_config",
            "megatron_core", "use_rslora", "use_dora", "layer_replication",
        }

        cleaned = {k: v for k, v in config.items() if k in valid_keys}
        with open(config_path, "w") as f:
            json.dump(cleaned, f, indent=2)

    def generate(self, lang: str, question: str, context: str = "", instruction: str = "") -> str:
        """Generate answer for given language and question.

        context: grounding material the answer should be based on (a
        retrieved reference, a document excerpt, reviewer feedback). The
        model treats this as source material to draw the answer from.

        instruction: a behavioral directive about HOW to answer (e.g.
        "attribute quoted facts explicitly") — kept in its own labeled
        section rather than folded into `context`. Mixing the two risks
        the model treating an instruction sentence as more legal content
        to ground the answer in, rather than as a directive to follow.
        """
        from config import LANG_NAMES, MAX_NEW_TOKENS

        self.load_adapter(lang)

        lang_name = LANG_NAMES.get(lang, "English")

        # ── Build prompt ──
        system_prompt = f"You are an expert Indian legal assistant. Answer the following legal question in {lang_name}."
        if context:
            system_prompt += f"\n\nRelevant context:\n{context}"
        if instruction:
            system_prompt += f"\n\nInstructions:\n{instruction}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text, return_tensors="pt"
        ).to(self.base_model.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return answer.strip()

    def unload(self):
        """Free GPU memory"""
        del self.base_model
        self.base_model = None
        self.current_adapter = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✅ Model unloaded")


# ── Singleton instance ──
model_loader = ModelLoader()