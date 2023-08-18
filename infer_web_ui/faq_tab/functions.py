import traceback

import gradio as gr


def load_faq(tab_faq):
    try:
        if tab_faq == "常见问题解答":
            with open("docs/faq.md", "r", encoding="utf8") as f:
                info = f.read()
        else:
            with open("docs/faq_en.md", "r", encoding="utf8") as f:
                info = f.read()
        return gr.Markdown(value=info)
    except Exception as e:
        print(e)
        return gr.Markdown(traceback.format_exc())
