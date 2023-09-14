import argparse
import json

from . import parse_document_xml


def main() -> None:  # pragma no cover
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GROBID TEI XML to Json and Mardown",
        usage="%(prog)s [options] <teifile>...",
    )
    parser.add_argument(
        "--no-encumbered",
        action="store_true",
        help="don't include ambiguously copyright encumbered fields (eg, abstract, body)",
    )
    parser.add_argument("teifiles", nargs="+")

    args = parser.parse_args()

    for filename in args.teifiles:
        content = open(filename, "r").read()
        doc = parse_document_xml(content)
        if args.no_encumbered:
            doc.remove_encumbered()
        print(json.dumps(doc.to_dict(), sort_keys=True))


if __name__ == "__main__":  # pragma no cover
    main()
