"""FAQ tab (renders docs/cn/faq.md or docs/en/faq_en.md)."""

import traceback

import gradio as gr

from web.runtime import i18n


def build():
    tab_faq = i18n("常见问题解答")
    with gr.TabItem(tab_faq):
        try:
            if tab_faq == "常见问题解答":
                with open("docs/cn/faq.md", "r", encoding="utf8") as f:
                    info = f.read()
            else:
                with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                    info = f.read()
            gr.Markdown(value=info)
        except:
            gr.Markdown(traceback.format_exc())
