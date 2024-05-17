from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

class LlamaForCausalRAG:
    
    def __init__(self,config,logger):
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.logger = logger
        use_GPU = config.get("acceleration", {}).get("useGPU")
        if(use_GPU):
            self.device = config.get("acceleration", {}).get("GPU")
            if self.device == "mps":
                logger.info(f"Load model with mps specific configuration")
                self.n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
                self.n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
                self.n_ctx = 4096
            if self.device == "cuda":
                logger.info(f"Load model with cuda specific configuration")
                self.n_gpu_layers = -1 # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
                self.n_batch = 1024 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                self.n_ctx = 4096
        else:
            self.device = "cpu"
    
       

    def load_llm(self,path):
       
        if self.device == "cpu":
            self.logger.info(f"Initialize LLM CPU configuration")
            return LlamaCpp(
                model_path=path, #it is important to quantisize the model in order to use it with llamaCpp check: https://colab.research.google.com/drive/1jeb9RoOVW984EpUAA_XNu1KfoyJOCe2Q#scrollTo=xBuDTDcIvIOQ
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                callback_manager=self.callback_manager,
                max_tokens=4096,
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
                max_tokens = 4096,
                temperature = 0.9,
                n_ctx = self.n_ctx, 
                verbose = True,  # Verbose is required to pass to the callback manager
            )

