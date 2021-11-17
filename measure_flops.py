# https://github.com/Lyken17/pytorch-OpCounter
from thop.profile import profile, clever_format

inputs = None
model = None
macs, params = profile(model, (inputs,), verbose=False)
macs, params = clever_format([macs, params], "%.3f")
