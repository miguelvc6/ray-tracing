import torch as t

device = t.device("cuda" if t.cuda.is_available() else "cpu")
