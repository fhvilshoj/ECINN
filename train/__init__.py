from .train import train

def add_arguments(parser): pass

def needs_config(): return True
def needs_model():  return True 
def needs_data():   return True

def fn(args, cfg, model, data):
    train(cfg, model, data)
