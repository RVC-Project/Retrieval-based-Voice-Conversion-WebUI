#!/bin/bash

tensorboard --logdir /app/logs --bind_all & python infer-web.py
