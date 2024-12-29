from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from functools import lru_cache
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
        future_to_index = {
            executor.submit(func, item, **kwargs): i for i, item in enumerate(inputs)
        }
        results = [None] * len(inputs)
        for future in tqdm(as_completed(future_to_index), total=len(inputs), desc=desc):
            results[future_to_index[future]] = future.result()
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
        future_to_index = {
            executor.submit(func, item, **kwargs): i for i, item in enumerate(inputs)
        }
        results = [None] * len(inputs)
        for future in tqdm(as_completed(future_to_index), total=len(inputs), desc=desc):
            results[future_to_index[future]] = future.result()
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


@lru_cache(maxsize=1)
def _safely_import_torch():
    """
    Open3D has an issue where it must be imported before torch. If Open3D is
    installed, this function will import Open3D before torch. Otherwise, it
    will return simply import and return torch.

    Use this function to import torch within camtools to handle the Open3D
    import order issue. That is, within camtools, we shall avoid `import torch`,
    and instead use `from camtools.backend import torch`. As torch is an
    optional dependency for camtools, this function will return None if torch
    is not available.

    Returns:
        module: The torch module if available, otherwise None.
    """
    try:
        __import__("open3d")
    except ImportError:
        pass

    try:
        _torch = __import__("torch")
        return _torch
    except ImportError:
        return None


_safe_torch = _safely_import_torch()


def is_jpg_path(path):
    return Path(path).suffix.lower() in [".jpg", ".jpeg"]


def is_png_path(path):
    return Path(path).suffix.lower() in [".png"]
