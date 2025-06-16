class COLORS:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ORANGE = '\x1B[38;5;216;4m'
    WARNING = '\033[93m'
    NICE = '\x1B[38;5;216;1m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

cprint = lambda phrase: print(f"{COLORS.GREEN}{phrase}{COLORS.ENDC}")

printGreen = lambda term: print(f"{COLORS.GREEN}{term}{COLORS.ENDC}")
printCyan = lambda term: print(f"{COLORS.CYAN}{term}{COLORS.ENDC}")
printWarn = lambda term: print(f"{COLORS.WARNING}{term}{COLORS.ENDC}")
printNice = lambda term: print(f"{COLORS.NICE}{term}{COLORS.ENDC}")
printBlue = lambda term: print(f"{COLORS.BLUE}{term}{COLORS.ENDC}")
printOrange = lambda term: print(f"{COLORS.ORANGE}{term}{COLORS.ENDC}")

def table_parameters(model, only_trainable=True):
    def get_max_len_name(model):
        names = [n for n, _ in model.named_parameters()]
        return len(max(names, key=len))
    s_l = get_max_len_name(model)
    sum_fn = lambda l: sum(i.numel() for i in l.parameters())
    printNice(f"{model.__class__.__name__:_^{s_l}}|{'Parameters':_^25}|{'Set To Train':_^25}".upper())
    for n, l in model.named_children():
        if len(list(l.parameters())) and next(l.parameters()).requires_grad is not None:
            printOrange(f"{n:.^{s_l}}|{sum_fn(l):.^25,}|{'layer':.^25}".upper())
            for n_i, p in l.named_parameters():
                n_it = f"{n}.{n_i}"
                if p.requires_grad == True:
                    printGreen(f"{n_it:.^{s_l}}|{p.numel():.^25,}|{'True':.^25}")
                else:
                    printBlue(f"{n_it:.^{s_l}}|{p.numel():.^25,}|{'False':.^25}")
        else:
            printWarn(f"{n:.^{s_l}}|{'NA':.^25}|{'NA':.^25}".upper())
