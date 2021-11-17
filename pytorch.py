# device setup for model, tensors, etc.
# both are equivalent
model.cuda()
model.to(device)

# enables built.in cudnn auto.tuner to find the best algorithm to use for your hardware
torch.backends.cudnn.benchmark = True