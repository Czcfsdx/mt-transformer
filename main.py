import argparse
import sys
from src.config import DEFAULT_BEAM_SIZE

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translator",
        description="EN→ZH Transformer translator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _ = sub.add_parser("train", help="Train the Transformer.")

    tr = sub.add_parser("translate", help="Translate a single English sentence.")
    _ = tr.add_argument(
        "-t",
        "--text",
        type=str,
        help="English sentence to translate."
    )
    _ = tr.add_argument(
        "-b",
        "--beam",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size; 1 means greedy decoding. Default is {DEFAULT_BEAM_SIZE}.",
    )

    ev = sub.add_parser("evaluate", help="Evaluate BLEU on the test set.")
    _ = ev.add_argument(
        "-b",
        "--beam",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size; 1 means greedy decoding. Default is {DEFAULT_BEAM_SIZE}.",
    )

    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    match args.command:
        case "train":
            from src.train import train as run_train
            run_train()
        case "evaluate":
            from src.evaluate import evaluate as run_evaluate
            beam = args.beam if args.beam > 1 else 1
            run_evaluate(beam_size=beam)
        case "translate":

            beam = args.beam if args.beam > 1 else 1
            if args.text is None:
                from src.utils import interactive_translate
                interactive_translate(beam=beam)
            else:
                from src.utils import translate_text
                try:
                    text = translate_text(args.text, beam=beam)
                    print(text)
                except ValueError as e:
                    print(f"[translate] {str(e)}", file=sys.stderr)
                    sys.exit(1)
        case _:
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
