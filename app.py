"""Credit to https://github.com/THUDM/ChatGLM2-6B/blob/main/web_demo.py
while mistakes are mine
"""
# pylint: disable=broad-exception-caught, redefined-outer-name, missing-function-docstring, missing-module-docstring, too-many-arguments, line-too-long, invalid-name, redefined-builtin, redefined-argument-from-local
# import gradio as gr

# model_name = "models/THUDM/chatglm2-6b-int4"
# gr.load(model_name).lauch()

# %%writefile demo-4bit.py

import os
import time
from textwrap import dedent

import gradio as gr
import mdtex2html
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

# fix timezone in Linux
os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

# model_name = "THUDM/chatglm2-6b"
model_name = "THUDM/chatglm2-6b-int4"

RETRY_FLAG = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()

# 4/8 bit
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).quantize(4).cuda()

has_cuda = torch.cuda.is_available()
# has_cuda = False  # force cpu

if has_cuda:
    model = (
        AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda().half()
    )  # 3.92G
else:
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True
    ).half()  # .float() .half().float()

model = model.eval()

_ = """Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(
    RETRY_FLAG, input, chatbot, max_length, top_p, temperature, history, past_key_values
):
    try:
        chatbot.append((parse_text(input), ""))
    except Exception as exc:
        logger.error(exc)
        logger.debug(f"{chatbot=}")
        _ = """
        if chatbot:
            chatbot[-1] = (parse_text(input), str(exc))
            yield chatbot, history, past_key_values
        # """
        yield chatbot, history, past_key_values

    for response, history, past_key_values in model.stream_chat(
        tokenizer,
        input,
        history,
        past_key_values=past_key_values,
        return_past_key_values=True,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    ):
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history, past_key_values


def trans_api(input, max_length=4096, top_p=0.8, temperature=0.2):
    if max_length < 10:
        max_length = 4096
    if top_p < 0.1 or top_p > 1:
        top_p = 0.85
    if temperature <= 0 or temperature > 1:
        temperature = 0.01
    try:
        res, _ = model.chat(
            tokenizer,
            input,
            history=[],
            past_key_values=None,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
        # logger.debug(f"{res=} \n{_=}")
    except Exception as exc:
        logger.error(f"{exc=}")
        res = str(exc)

    return res


def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], [], None


# Delete last turn
def delete_last_turn(chat, history):
    if chat and history:
        chat.pop(-1)
        history.pop(-1)
    return chat, history


# Regenerate response
def retry_last_answer(
    user_input, chatbot, max_length, top_p, temperature, history, past_key_values
):
    if chatbot and history:
        # Removing the previous conversation from chat
        chatbot.pop(-1)
        # Setting up a flag to capture a retry
        RETRY_FLAG = True
        # Getting last message from user
        user_input = history[-1][0]
        # Removing bot response from the history
        history.pop(-1)

    yield from predict(
        RETRY_FLAG,  # type: ignore
        user_input,
        chatbot,
        max_length,
        top_p,
        temperature,
        history,
        past_key_values,
    )


with gr.Blocks(title="ChatGLM2-6B-int4", theme=gr.themes.Soft(text_size="sm")) as demo:
    # gr.HTML("""<h1 align="center">ChatGLM2-6B-int4</h1>""")
    gr.HTML(
        """<center><a href="https://huggingface.co/spaces/mikeee/chatglm2-6b-4bit?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>To avoid the queue and for faster inference Duplicate this Space and upgrade to GPU</center>"""
    )

    with gr.Accordion("🎈 Info", open=False):
        _ = f"""
            ## {model_name}

            Try to refresh the browser and try again when  occasionally an error occurs.

            With a GPU, a query takes from a few seconds to a few tens of seconds, dependent on the number of words/characters
            the question and responses contain. The quality of the responses varies quite a bit it seems. Even the same
            question with the same parameters, asked at different times, can result in quite different responses.

            * Low temperature: responses will be more deterministic and focused; High temperature: responses more creative.

            * Suggested temperatures -- translation: up to 0.3; chatting: > 0.4

            * Top P controls dynamic vocabulary selection based on context.

            For a table of example values for different scenarios, refer to [this](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683)

            If the instance is not on a GPU (T4), it will be very slow. You can try to run the colab notebook [chatglm2-6b-4bit colab notebook](https://colab.research.google.com/drive/1WkF7kOjVCcBBatDHjaGkuJHnPdMWNtbW?usp=sharing) for a spin.

            The T4 GPU is sponsored by a community GPU grant from Huggingface. Thanks a lot!
            """
        gr.Markdown(dedent(_))
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Input...",
                ).style(container=False)
                RETRY_FLAG = gr.Checkbox(value=False, visible=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row():
                    submitBtn = gr.Button("Submit", variant="primary")
                    deleteBtn = gr.Button("Delete last turn", variant="secondary")
                    retryBtn = gr.Button("Regenerate", variant="secondary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0,
                32768,
                value=8192,
                step=1.0,
                label="Maximum length",
                interactive=True,
            )
            top_p = gr.Slider(
                0, 1, value=0.85, step=0.01, label="Top P", interactive=True
            )
            temperature = gr.Slider(
                0.01, 1, value=0.95, step=0.01, label="Temperature", interactive=True
            )

    history = gr.State([])
    past_key_values = gr.State(None)

    user_input.submit(
        predict,
        [
            RETRY_FLAG,
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        [chatbot, history, past_key_values],
        show_progress="full",
    )
    submitBtn.click(
        predict,
        [
            RETRY_FLAG,
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        [chatbot, history, past_key_values],
        show_progress="full",
        api_name="predict",
    )
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(
        reset_state, outputs=[chatbot, history, past_key_values], show_progress="full"
    )

    retryBtn.click(
        retry_last_answer,
        inputs=[
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        # outputs = [chatbot, history, last_user_message, user_message]
        outputs=[chatbot, history, past_key_values],
    )
    deleteBtn.click(delete_last_turn, [chatbot, history], [chatbot, history])

    with gr.Accordion("Example inputs", open=True):
        etext = """In America, where cars are an important part of the national psyche, a decade ago people had suddenly started to drive less, which had not happened since the oil shocks of the 1970s. """
        examples = gr.Examples(
            examples=[
                ["Explain the plot of Cinderella in a sentence."],
                [
                    "How long does it take to become proficient in French, and what are the best methods for retaining information?"
                ],
                ["What are some common mistakes to avoid when writing code?"],
                ["Build a prompt to generate a beautiful portrait of a horse"],
                ["Suggest four metaphors to describe the benefits of AI"],
                ["Write a pop song about leaving home for the sandy beaches."],
                ["Write a summary demonstrating my ability to tame lions"],
                ["鲁迅和周树人什么关系"],
                ["从前有一头牛，这头牛后面有什么？"],
                ["正无穷大加一大于正无穷大吗？"],
                ["正无穷大加正无穷大大于正无穷大吗？"],
                ["-2的平方根等于什么"],
                ["树上有5只鸟，猎人开枪打死了一只。树上还有几只鸟？"],
                ["树上有11只鸟，猎人开枪打死了一只。树上还有几只鸟？提示：需考虑鸟可能受惊吓飞走。"],
                ["鲁迅和周树人什么关系 用英文回答"],
                ["以红楼梦的行文风格写一张委婉的请假条。不少于320字。"],
                [f"{etext} 翻成中文，列出3个版本"],
                [f"{etext} \n 翻成中文，保留原意，但使用文学性的语言。不要写解释。列出3个版本"],
                ["js 判断一个数是不是质数"],
                ["js 实现python 的 range(10)"],
                ["js 实现python 的 [*(range(10)]"],
                ["假定 1 + 2 = 4, 试求 7 + 8"],
                ["Erkläre die Handlung von Cinderella in einem Satz."],
                ["Erkläre die Handlung von Cinderella in einem Satz. Auf Deutsch"],
            ],
            inputs=[user_input],
            examples_per_page=30,
        )

    with gr.Accordion("For Chat/Translation API", open=False, visible=False):
        input_text = gr.Text()
        tr_btn = gr.Button("Go", variant="primary")
        out_text = gr.Text()
    tr_btn.click(
        trans_api,
        [input_text, max_length, top_p, temperature],
        out_text,
        # show_progress="full",
        api_name="tr",
    )
    _ = """
    input_text.submit(
        trans_api,
        [input_text, max_length, top_p, temperature],
        out_text,
        show_progress="full",
        api_name="tr1",
    )
    # """

# demo.queue().launch(share=False, inbrowser=True)
# demo.queue().launch(share=True, inbrowser=True, debug=True)

demo.queue().launch(debug=True)
