import sys
from pathlib import Path
import camtools as ct
import argparse


def _print_greetings():
    greeting_str = f"* CamTools: Camera Tools for Computer Vision (v{ct.__version__}) *"
    header = "*" * len(greeting_str)
    print(header)
    print(greeting_str)
    print(header)


def main():
    _print_greetings()

    main_parser = argparse.ArgumentParser(
        description="CamTools CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub_parsers = main_parser.add_subparsers(
        title="subcommand",
        dest="subcommand",
        help="Select one of these commands.\n\n",
    )

    # ct crop-boarders
    sub_parser = sub_parsers.add_parser(
        "crop-boarders",
        help="Crop boarders of images.\n"
        "```\n"
        "ct crop-boarders *.png --pad_pixel 10 --skip_cropped --same_crop\n"
        "```",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub_parser = ct.tools.crop_boarders.instantiate_parser(sub_parser)
    sub_parser.set_defaults(func=ct.tools.crop_boarders.entry_point)

    # ct draw-bboxes a.png b.png
    sub_parser = sub_parsers.add_parser(
        "draw-bboxes",
        help="Draw bounding boxes on images.\n"
        "```\n"
        "ct draw-bboxes path/to/a.png\n"
        "ct draw-bboxes path/to/a.png path/to/b.png\n"
        "```",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub_parser = ct.tools.draw_bboxes.instantiate_parser(sub_parser)
    sub_parser.set_defaults(func=ct.tools.draw_bboxes.entry_point)

    args = main_parser.parse_args()
    if args.subcommand in sub_parsers.choices.keys():
        return args.func(sub_parsers.choices[args.subcommand], args)
    else:
        main_parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
