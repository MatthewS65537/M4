def count_params(model, trainable=None):
    if trainable == None:
        return sum(p.numel() for p in model.parameters())
    elif trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif not trainable:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)
        