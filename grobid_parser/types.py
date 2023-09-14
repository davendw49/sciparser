from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GrobidAddress:
	addr_line: Optional[str] = None
	post_code: Optional[str] = None
	settlement: Optional[str] = None
	country: Optional[str] = None


@dataclass
class GrobidAffiliation:
	institution: Optional[str] = None
	department: Optional[str] = None
	laboratory: Optional[str] = None
	address: Optional[GrobidAddress] = None


@dataclass
class GrobidAuthor:
	full_name: Optional[str]
	given_name: Optional[str] = None
	middle_name: Optional[str] = None
	surname: Optional[str] = None
	email: Optional[str] = None  # TODO: test coverage
	orcid: Optional[str] = None  # TODO: test coverage
	affiliation: Optional[GrobidAffiliation] = None

	def to_csl_dict(self) -> dict:
		d = dict(
			given=self.given_name or self.middle_name,
			family=self.surname,
		)
		return _simplify_dict(d)


def _csl_date(s: Optional[str]) -> Optional[list]:
	if not s:
		return None

	# YYYY
	if len(s) >= 4 and s[0:4].isdigit():
		year = int(s[0:4])
	else:
		return None

	# YYYY-MM
	if len(s) >= 7 and s[4] == "-" and s[5:7].isdigit():
		month = int(s[5:7])
	else:
		return [[year]]

	# YYYY-MM-DD
	if len(s) == 10 and s[7] == "-" and s[8:10].isdigit():
		day = int(s[8:10])
		return [[year, month, day]]
	else:
		return [[year, month]]


def test_csl_date() -> None:
	assert _csl_date("1998") == [[1998]]
	assert _csl_date("1998-03") == [[1998, 3]]
	assert _csl_date("1998-03-12") == [[1998, 3, 12]]
	assert _csl_date("1998-blah") == [[1998]]
	assert _csl_date("asdf") is None


@dataclass
class GrobidBiblio:
	authors: List[GrobidAuthor]
	index: Optional[int] = None
	id: Optional[str] = None
	unstructured: Optional[str] = None

	date: Optional[str] = None
	title: Optional[str] = None
	book_title: Optional[str] = None
	series_title: Optional[str] = None
	editors: Optional[List[GrobidAuthor]] = None
	journal: Optional[str] = None
	journal_abbrev: Optional[str] = None
	publisher: Optional[str] = None
	institution: Optional[str] = None
	issn: Optional[str] = None
	eissn: Optional[str] = None
	volume: Optional[str] = None
	issue: Optional[str] = None
	pages: Optional[str] = None
	first_page: Optional[str] = None
	last_page: Optional[str] = None
	note: Optional[str] = None

	doi: Optional[str] = None
	pmid: Optional[str] = None
	pmcid: Optional[str] = None
	arxiv_id: Optional[str] = None
	pii: Optional[str] = None
	ark: Optional[str] = None
	istex_id: Optional[str] = None
	url: Optional[str] = None

	def is_empty(self) -> bool:
		"""
		If true, this record is empty (contains no actual metadata)
		"""
		if self.authors or self.editors:
			return False
		return not bool(
			self.date
			or self.title
			or self.journal
			or self.publisher
			or self.volume
			or self.issue
			or self.pages
			or self.doi
			or self.pmid
			or self.pmcid
			or self.arxiv_id
			or self.url
		)

	def to_dict(self) -> dict:
		return _simplify_dict(asdict(self))

	def to_legacy_dict(self) -> dict:
		"""
		Returns a dict in the old "grobid2json" format.
		"""
		d = self.to_dict()

		# new keys
		d.pop("first_page", None)
		d.pop("last_page", None)
		d.pop("note", None)

		# legacy book title behavior
		if not d.get("journal") and d.get("book_title"):
			d["journal"] = d.pop("book_title")
		else:
			d.pop("book_title", None)

		# author changes
		for a in d["authors"]:
			a["name"] = a.pop("full_name", None)
			if not a.get("given_name"):
				a["given_name"] = a.pop("middle_name", None)
			else:
				a.pop("middle_name", None)
			addr = a.get("affiliation", {}).get("address")
			if addr and addr.get("post_code"):
				addr["postCode"] = addr.pop("post_code")

		return _simplify_dict(d)

	def to_csl_dict(self, default_type: str = "article-journal") -> dict:
		"""
		Transforms in to Citation Style Language (CSL) JSON schema, as a dict
		(not an actual JSON string)
		"""
		csl: Dict[str, Any] = dict(
			type=default_type,
			author=[a.to_csl_dict() for a in self.authors],
			issued=_csl_date(self.date),
			publisher=self.publisher,
			title=self.title,
			page=self.pages,
			URL=self.url,
			DOI=self.doi,
			PMID=self.pmid,
			PMCID=self.pmcid,
			ISSN=self.issn,
			note=self.note,
		)
		# fields with '-' in the key name
		csl.update(
			{
				"container-title": self.journal,
				"book-title": self.book_title,
				"series-title": self.series_title,
				"page-first": self.first_page,
			}
		)

		# numeric fields
		if self.issue and self.issue.isdigit():
			csl["issue"] = int(self.issue)
		if self.volume and self.volume.isdigit():
			csl["volume"] = int(self.volume)

		return _simplify_dict(csl)


class GrobidBody:
	text: Optional[str] = None


class GrobidTable:
	text: Optional[str] = None


class GrobidFigure:
	figure_type: Optional[str] = None
	figure_lable: Optional[str] = None
	text: Optional[str] = None
	figure_id: Optional[str] = None
	figure_schema_type: Optional[str] = None


class GrobidNote:
	text: Optional[str] = None


@dataclass
class GrobidDocument:
	grobid_version: str
	grobid_timestamp: str
	header: GrobidBiblio

	pdf_md5: Optional[str] = None
	language_code: Optional[str] = None
	citations: Optional[List[GrobidBiblio]] = None
	abstract: Optional[str] = None
	body: Optional[str] = None
	acknowledgement: Optional[str] = None
	annex: Optional[str] = None

	def to_dict(self) -> dict:
		"""
		Returns a dict version of this object which has no 'None' fields
		(recursively), and is appropriate for serializing to JSON with
		json.dumps().

		If you did want all the fields, you could use dataclasses.asdict()
		directly on thing object.
		"""
		return _simplify_dict(asdict(self))

	def to_legacy_dict(self) -> dict:
		"""
		Returns a dict in the old "grobid2json" format.
		"""
		d = self.to_dict()
		d.pop("header", None)
		d.update(self.header.to_legacy_dict())
		if self.citations:
			d["citations"] = [c.to_legacy_dict() for c in self.citations]

		# all header fields at top-level
		d["journal"] = dict(
			name=d.pop("journal", None),
			publisher=d.pop("publisher", None),
			issn=d.pop("issn", None),
			issne=d.pop("issne", None),
			volume=d.pop("volume", None),
			issue=d.pop("issue", None),
		)

		# document fields not in the old schema
		d.pop("pdf_md5", None)

		return _simplify_dict(d)

	def remove_encumbered(self) -> None:
		"""
		This helper function removes fields from this object which might raise
		copyright concerns.
		"""
		self.abstract = None
		self.body = None
		self.acknowledgement = None
		self.annex = None

	def to_csl_dict(self, default_type: str = "article-journal") -> dict:
		"""
		Transforms in to Citation Style Language (CSL) JSON schema, as a dict
		(not an actual JSON string)
		"""
		return self.header.to_csl_dict(default_type=default_type)


def _simplify_dict(d: dict) -> dict:
	"""
	Recursively remove empty dict values from a dict and all sub-lists and
	sub-dicts.

	TODO: should this return Optional[dict]?
	"""
	if d in [None, {}, ""]:
		return {}
	for k in list(d.keys()):
		if isinstance(d[k], dict):
			d[k] = _simplify_dict(d[k])
		elif isinstance(d[k], list):
			for i in range(len(d[k])):
				if isinstance(d[k][i], dict):
					d[k][i] = _simplify_dict(d[k][i])
		if d[k] in [None, {}, ""]:
			d.pop(k)
	return d
