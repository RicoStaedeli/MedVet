from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

class LlamaForCausalRAG:
    '''
    This class is able to load a model from a GGUF file. 
    GGUF models can be loaded with an Apple Silicon M2 or M3
    This is the default to load a model and should not be changed.
    The class loads the model for the available GPU device. The device has to be set in the cfg.yaml file
    '''
    def __init__(self,config,logger):
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.logger = logger
        use_GPU = config.get("acceleration", {}).get("useGPU")
        self.temperature = config.get("LLM", {}).get("temperature")
        self.max_tokens = config.get("LLM", {}).get("max_tokens")
        self.n_batch = config.get("LLM", {}).get("n_batch")
        self.n_ctx = config.get("LLM", {}).get("n_ctx")
        if(use_GPU):
            self.device = config.get("acceleration", {}).get("GPU")
            if self.device == "mps":
                logger.info(f"Load model with mps specific configuration")
                self.n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
                self.n_batch = self.n_batch  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
                self.n_ctx = self.n_ctx
            if self.device == "cuda":
                logger.info(f"Load model with cuda specific configuration")
                self.n_gpu_layers = -1 # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
                self.n_batch = self.n_batch # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                self.n_ctx = self.n_ctx
        else:
            self.device = "cpu"
    
       

    def load_llm(self,path):
       
        if self.device == "cpu":
            self.logger.info(f"Initialize LLM CPU configuration")
            return LlamaCpp(
                model_path=path, #it is important to quantisize the model in order to use it with llamaCpp check: https://colab.research.google.com/drive/1jeb9RoOVW984EpUAA_XNu1KfoyJOCe2Q#scrollTo=xBuDTDcIvIOQ
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                callback_manager=self.callback_manager,
                max_tokens=self.max_tokens,
                verbose=True,  # Verbose is required to pass to the callback manager
            )
        else:
            self.logger.info(f"Initialize LLM GPU configuration")
            return LlamaCpp(
                model_path = path, #it is important to quantisize the model in order to use it with llamaCpp check: https://colab.research.google.com/drive/1jeb9RoOVW984EpUAA_XNu1KfoyJOCe2Q#scrollTo=xBuDTDcIvIOQ
                n_gpu_layers = self.n_gpu_layers,
                n_batch = self.n_batch,
                f16_kv = True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                callback_manager = self.callback_manager,
                max_tokens = self.max_tokens,
                temperature = self.temperature,
                n_ctx = self.n_ctx, 
                verbose = True,  # Verbose is required to pass to the callback manager
            )

