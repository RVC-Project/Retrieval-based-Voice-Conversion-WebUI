#!/bin/bash

pip install tensorflow && tensorboard --logdir /app/logs --bind_all & python infer-web.py
