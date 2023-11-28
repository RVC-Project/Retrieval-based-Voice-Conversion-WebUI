#!/bin/bash

load_ext tensorboard & tensorboard --logdir /app/logs --bind_all & python infer-web.py
