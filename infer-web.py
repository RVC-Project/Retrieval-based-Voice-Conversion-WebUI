import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from web.app import build_app
from web.runtime import config

app = build_app()

if config.iscolab:
    app.queue(concurrency_count=511, max_size=1022).launch(share=True)
else:
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=config.listen_port,
        quiet=True,
    )
