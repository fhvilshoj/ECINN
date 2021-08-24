from .compute_and_store import compute_counterfactuals

def add_arguments(parser):
    parser.add_argument('--max_samples', '-ms',         type=int, default=-1, help="Maximum number of samples to compute counterfactuals for.")

def needs_config(): return True
def needs_model():  return True 
def needs_data():   return True

def fn(args, cfg, model, data):
    print('# # ' * 15)
    print('# %-55s #' % "Computing counterfactuals")
    print('# # ' * 15)
    compute_counterfactuals(args, cfg, model, data)

