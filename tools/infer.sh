#!/bin/bash

tensorboard --logdir /workspace/logs --bind_all & python infer-web.py
