from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from tqdm import tqdm


def mt_loop(
    func: Callable[[Any], Any],
    inputs: Iterable[Any],
    **kwargs,
) -> list:
    """
    Applies a function to each item in the given list in parallel using multi-threading.

    Args:
        func (Callable[[Any], Any]): The function to apply. Must accept a single
            argument.
        inputs (Iterable[Any]): An iterable of inputs to process with the
            function.
        **kwargs: Additional keyword arguments to pass to `func`.

    Returns:
        list: A list of results from applying `func` to each item in `list_input`.
    """
    desc = f"[mt] {func.__name__}"
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, item, **kwargs) for item in inputs]
        progress = tqdm(
            as_completed(futures),
            total=len(inputs),
            desc=desc,
        )
        results = [future.result() for future in progress]
    return results


def mp_loop(
    func: Callable[[Any], Any],
    inputs: Iterable[Any],
    **kwargs,
) -> list:
    """
    Applies a function to each item in the given list in parallel using multi-processing.

    Args:
        func (Callable[[Any], Any]): The function to apply. Must accept a single
            argument.
        inputs (Iterable[Any]): An iterable of inputs to process with the
            function.
        **kwargs: Additional keyword arguments to pass to `func`.

    Returns:
        list: A list of results from applying `func` to each item in `inputs`.
    """
    desc = f"[mp] {func.__name__}"
    with ProcessPoolExecutor() as executor:
        future_to_item = {
            executor.submit(func, item, **kwargs): item for item in inputs
        }
        progress = tqdm(
            as_completed(future_to_item),
            total=len(inputs),
            desc=desc,
        )
        results = [future.result() for future in progress]
    return results


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
        ```python
        if query_yes_no("Continue?", default="yes"):
            print("Proceeding.")
        else:
            print("Aborted.")
        ```

        ```python
        if not query_yes_no("Continue?", default="yes"):
            print("Aborted.")
            return  # Or exit(0)
        print("Proceeding.")
        ```
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
