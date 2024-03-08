from dotenv import load_dotenv
load_dotenv(verbose=False)

import os

import gradio as gr

from generate import Generator

DEMO_PORT = int(os.getenv("DEMO_PORT"))

generator = Generator(model="chatopenai")

def predict(message, history=[]):
    response = generator.generate(message)
    return response

gr.ChatInterface(title="캐릭터 설정 생성기",
                 fn = predict,
                 description="로그라인을 기반으로 캐릭터를 생성하는 페이지입니다. 아래의 예시를 참고하여 로그라인을 입력해주세요.",
                 examples=[['하고 싶은 것도 많고 짝사랑 선배와 사랑까지 이루고 싶은 18세 소녀 ‘유미’. 하지만 심장이 빨리 뛰면 폭발해버리는 병에 걸린 동급생 ‘재오’와 정략 결혼을 당해버린다. 20살이 되는 즉시 결혼하고 애까지 낳아야 하는 유미의 미래! 좌절에 빠진 유미는 ‘약 혼남이 없어진다면 결혼 안 해도 되는 거 아니야?’ 라는 생각이 들고 재오가 성인이 되기 전까지 심장을 빨리 뛰게 만들어서 죽일 계획을 세운다.']],
                 theme="soft",
                 retry_btn="다시보내기 ↩",
                 undo_btn="이전챗 삭제 ❌",
                 clear_btn="전챗 삭제 💫"
                 ).launch(share=True, server_port=DEMO_PORT)

