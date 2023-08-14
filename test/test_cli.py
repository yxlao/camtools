import subprocess


def _run_and_test_command(cmd: str, required_stdout: str = None):
    """
    Run cmd in terminal, assert return code is 0 and check for required output.

    Args:
        cmd (str): Command to run in terminal.
        required_stdout (str): Required std output from command.
    """
    # Run command.
    cmd_tokens = cmd.split()
    proc = subprocess.run(cmd_tokens, capture_output=True, check=False)

    # Assert return code is 0.
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd_tokens)

    # Assert required output is in stdout.
    if required_stdout is not None:
        std_out = proc.stdout.decode("utf-8")
        if required_stdout not in std_out:
            raise AssertionError(
                f"Required stdout not found in output of {cmd}: {std_out}"
            )


def test_ct():
    _run_and_test_command(
        cmd="ct --help",
        required_stdout="usage: ct",
    )


def test_crop_boarders():
    _run_and_test_command(
        cmd="ct crop-boarders --help",
        required_stdout="usage: ct crop-boarders",
    )


def test_draw_bboxes():
    _run_and_test_command(
        cmd="ct draw-bboxes --help",
        required_stdout="usage: ct draw-bboxes",
    )
