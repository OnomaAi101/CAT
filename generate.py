
from dotenv import load_dotenv
import os
load_dotenv(verbose=False)


from typing import Union, List, Dict

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

RETRIEVAL_ARGS = {
    "search_type": "similarity",
    "search_kwargs": {"k": 3}
}

LLM_ARGS = {
    "model_name": "gpt-4-turbo",
    "azure_deployment": os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),
}

EMBEDDING_ARGS = {
    "azure_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    }

OPENAI_ARGS = {
    "model_name": "gpt-4-0125-preview",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "max_tokens": 1024,
}

# /data8/langchain/logline_character/main_character.txt 파일을 읽어서 Template으로 사용
with open("/data8/langchain/logline_character/main_character.txt", "r") as f:
    TEMPLATE = f.read()

class Generator:
    def __init__(self, 
                #  vectorstore_path:str = "./chroma_db",
                 model:str = "azurechatopenai",
                 temperature:float = 0.2):
        
        self.model = model
        
        # self.vectorstore = Chroma(
        #     embedding_function=AzureOpenAIEmbeddings(**EMBEDDING_ARGS), 
        #     persist_directory=vectorstore_path)
        # self.retriever = self.vectorstore.as_retriever(**RETRIEVAL_ARGS)
        
        if self.model == "azurechatopenai":
            self.llm = AzureChatOpenAI(**LLM_ARGS,temperature=temperature)
        elif self.model == "chatopenai":
            self.llm = ChatOpenAI(**OPENAI_ARGS, temperature=temperature)
        else:
            self.llm = HuggingFacePipeline.from_model_id(
                        model_id=model, 
                        device_map="auto",
                        task="text-generation", # 텍스트 생성
                        model_kwargs={"temperature": 0.1, 
                                    "do_sample": True},
                        pipeline_kwargs={"max_new_tokens": 256}
)
            
        print(f"Model Loaded : {self.llm}")
            
        self.template = PromptTemplate.from_template(TEMPLATE)
        
        self.chain = self.template | self.llm

    def generate(self, keyword) -> Union[str, List[Union[str, Dict]]]:
        

        if self.model == "chatopenai":
            result = self.chain.invoke({"logline": keyword}).content
            return result
        else:
            return self.chain.invoke({"logline": keyword})


if __name__ == "__main__":

    generator = Generator()

    message = "하고 싶은 것도 많고 짝사랑 선배와 사랑까지 이루고 싶은 18세 소녀 ‘유미’. 하지만 심장이 빨리 뛰면 폭발해버리는 병에 걸린 동급생 ‘재오’와 정략 결혼을 당해버린다. 20살이 되는 즉시 결혼하고 애까지 낳아야 하는 유미의 미래! 좌절에 빠진 유미는 ‘약 혼남이 없어진다면 결혼 안 해도 되는 거 아니야?’ 라는 생각이 들고 재오가 성인이 되기 전까지 심장을 빨리 뛰게 만들어서 죽일 계획을 세운다."

    response = generator.generate(message)

    print(response)