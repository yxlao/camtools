def query_yes_no(question, default=None):
    """Ask a yes/no question via raw_input() and return their answer.

    Args:
        question: A string that is presented to the user.
        default: The presumed answer if the user just hits <Enter>.
            - True: The answer is assumed to be yes.
            - False: The answer is assumed to be no.
            - None: The answer is required from the user.

    Returns:
        Returns True for "yes" or False for "no".

    Examples:
        if query_yes_no("Continue?", default="yes"):
            print("Proceeding.")
        else:
            print("Aborted.")
    """
    if default is None:
        prompt = "[y/n]"
    elif default == True:
        prompt = "[Y/n]"
    elif default == False:
        prompt = "[y/N]"
    else:
        raise ValueError(f"Invalid default answer: '{default}'")

    response_to_bool = {
        "yes": True,
        "y": True,
        "ye": True,
        "no": False,
        "n": False,
        True: True,
        False: False,
    }
    while True:
        print(f"{question} {prompt} ", end="")
        choice = input().lower()
        if default is not None and choice == "":
            return response_to_bool[default]
        elif choice in response_to_bool:
            return response_to_bool[choice]
        else:
            print('Please respond with "yes" or "no" (or "y" or "n").')
