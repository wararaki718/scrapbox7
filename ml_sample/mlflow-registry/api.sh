#!/bin/bash

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow models serve -m "models:/sk-learn-random-forest-reg-model/1"
