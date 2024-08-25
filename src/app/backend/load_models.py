from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from peft import (
    get_peft_model,
    PromptTuningConfig,
    TaskType,
    PromptTuningInit,
    LoraConfig,
    PeftConfig,
    PeftModel
)
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from backend.configs import (HF_TOEKN, MAX_GENERATION_NEW_TOKENS, GENERATION_TEMP, DEVICE)

os.environ["HF_TOKEN"] = HF_TOEKN


def get_base_llms(model_path:str):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config= nf4_config if torch.cuda.is_available() else None,
            device_map=DEVICE,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generation_pipeline = pipeline(
            model=base_model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=GENERATION_TEMP,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=MAX_GENERATION_NEW_TOKENS,
        )
    return HuggingFacePipeline(pipeline=generation_pipeline)

def get_lora_and_prefix_llms(prefix_params_path: str, lora_params_path: str):

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    prefix_llm = None
    if prefix_params_path:
        PREFIX_PARAMS_SAVE_PATH = prefix_params_path  # "./models/prefix"
        prefix_config = PeftConfig.from_pretrained(PREFIX_PARAMS_SAVE_PATH)
        ## Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            prefix_config.base_model_name_or_path,
            trust_remote_code=True,
            quantization_config=nf4_config if torch.cuda.is_available() else None,
            device_map=DEVICE,
        )
        tokenizer = AutoTokenizer.from_pretrained(prefix_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        ## Add the prefix parameters to the base model
        prefix_model = PeftModel.from_pretrained(base_model, PREFIX_PARAMS_SAVE_PATH)
        prefix_generation_pipeline = pipeline(
            model=prefix_model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=GENERATION_TEMP,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=MAX_GENERATION_NEW_TOKENS,
        )
        prefix_llm = HuggingFacePipeline(pipeline=prefix_generation_pipeline)

    ############# ----------- LORA ---------- ##############
    lora_llm = None
    if lora_params_path:
        LORA_PARAMS_SAVE_PATH = lora_params_path  # "../models/lora"
        lora_configs = LoraConfig.from_pretrained(LORA_PARAMS_SAVE_PATH)

        base_model = AutoModelForCausalLM.from_pretrained(
            lora_configs.base_model_name_or_path,
            trust_remote_code=True,
            quantization_config=nf4_config if torch.cuda.is_available() else None,
            device_map=DEVICE,
        )
        tokenizer = AutoTokenizer.from_pretrained(lora_configs.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        lora_model = get_peft_model(base_model, lora_configs)
        lora_generation_pipeline = pipeline(
            model=lora_model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=GENERATION_TEMP,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=MAX_GENERATION_NEW_TOKENS,
        )
        lora_llm = HuggingFacePipeline(pipeline=lora_generation_pipeline)

    return prefix_llm, lora_llm


def get_llm_pipeline(model_path_or_name:str):


    return get_base_llms(model_path_or_name)