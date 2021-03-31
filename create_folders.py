import os

if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("config"):
    os.makedirs("config")

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")
