class TerminalColours:
    header = '\033[95m'
    okblue = '\033[94m'
    okcyan = '\033[96m'
    okgreen = '\033[92m'
    warning = '\033[93m'
    fail = '\033[91m'
    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'


YAML_NECESSARY_VARIABLES: list[str] = [
    'database',
    'features_continuous',
    'features_categorical',
]