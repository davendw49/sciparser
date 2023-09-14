__version__ = "1.0.0"

from .parse import (
    parse_citation_list_xml,
    parse_citation_xml,
    parse_citations_xml,
    parse_document_xml,
)
from .types import GrobidBiblio, GrobidDocument
