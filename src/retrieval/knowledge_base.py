"""Construction of the commonsense knowledge base.

The retrieval layer needs a corpus of short, natural-language *facts*. This
module turns three real knowledge sources into that corpus. It never invents a
fact: every emitted string is a verbalisation of a tuple that exists in the
source knowledge graph.

Supported sources
-----------------
``conceptnet``
    The official ConceptNet 5.7 assertions dump. English-to-English edges above
    a weight threshold are kept, restricted to a sarcasm-relevant relation
    whitelist, and verbalised with the standard relation templates used in the
    commonsense-reasoning literature (Speer et al., 2017; the same templates
    used by KagNet/MHGRN-style systems).

``atomic``
    COMET-style inferential tuples (ATOMIC / ATOMIC-2020 TSV). These are the
    "COMET-derived evidence" of the proposal: they describe intents, reactions
    and effects, which is exactly the knowledge needed to notice that a
    positive-sounding utterance describes an undesirable situation.

``wordnet``
    WordNet 3.0 + SentiWordNet, via NLTK. Glosses, hypernyms, antonyms and
    sentiment polarity. WordNet is small, permissively licensed and downloadable
    without credentials, which makes it the default *offline-capable* source and
    the basis of the smoke-test sample index.

Every emitted fact keeps its provenance (source, relation, head, tail, weight)
so the explanation layer can cite it honestly.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.utils.paths import resolve

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- schema


@dataclass(slots=True)
class KnowledgeFact:
    """One natural-language commonsense fact with full provenance.

    Attributes:
        id: Stable identifier, unique within the knowledge base.
        text: The verbalised fact, e.g. ``"a traffic jam causes frustration"``.
        source: ``"ConceptNet"`` / ``"ATOMIC"`` / ``"WordNet"``.
        relation: Original relation name, e.g. ``"Causes"``, ``"xReact"``.
        head: Left-hand concept of the tuple.
        tail: Right-hand concept of the tuple.
        weight: Source-provided confidence (ConceptNet edge weight, 1.0 when the
            source has no notion of weight).
        metadata: Anything else worth keeping (dataset id, part of speech, ...).
    """

    id: str
    text: str
    source: str
    relation: str
    head: str = ""
    tail: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KnowledgeFact":
        """Rebuild from its dict form."""
        return cls(
            id=str(raw["id"]),
            text=str(raw["text"]),
            source=str(raw.get("source", "unknown")),
            relation=str(raw.get("relation", "")),
            head=str(raw.get("head", "")),
            tail=str(raw.get("tail", "")),
            weight=float(raw.get("weight", 1.0)),
            metadata=dict(raw.get("metadata") or {}),
        )


# ------------------------------------------------------------------ ConceptNet

#: Relations kept by default. Chosen because they express *causal, evaluative and
#: contrastive* knowledge -- the kinds that reveal the incongruity underlying
#: sarcasm. Purely taxonomic noise (``ExternalURL``, ``dbpedia/*``) is dropped.
CONCEPTNET_RELATIONS: dict[str, str] = {
    "IsA": "{head} is a type of {tail}",
    "HasProperty": "{head} is {tail}",
    "NotHasProperty": "{head} is not {tail}",
    "Causes": "{head} causes {tail}",
    "CausesDesire": "{head} makes someone want {tail}",
    "HasSubevent": "{tail} happens while {head}",
    "HasFirstSubevent": "{head} begins with {tail}",
    "HasLastSubevent": "{head} ends with {tail}",
    "HasPrerequisite": "{head} requires {tail}",
    "MotivatedByGoal": "someone does {head} because they want {tail}",
    "ObstructedBy": "{head} can be prevented by {tail}",
    "Desires": "{head} wants {tail}",
    "NotDesires": "{head} does not want {tail}",
    "CapableOf": "{head} can {tail}",
    "NotCapableOf": "{head} cannot {tail}",
    "UsedFor": "{head} is used for {tail}",
    "AtLocation": "{head} is found at {tail}",
    "PartOf": "{head} is part of {tail}",
    "MadeOf": "{head} is made of {tail}",
    "ReceivesAction": "{head} can be {tail}",
    "Antonym": "{head} is the opposite of {tail}",
    "DistinctFrom": "{head} is distinct from {tail}",
    "SimilarTo": "{head} is similar to {tail}",
    "Synonym": "{head} means the same as {tail}",
    "DefinedAs": "{head} is defined as {tail}",
    "MannerOf": "{head} is a way of {tail}",
    "SymbolOf": "{head} is a symbol of {tail}",
    "LocatedNear": "{head} is found near {tail}",
    "CreatedBy": "{head} is created by {tail}",
}

_CN_EN_CONCEPT = re.compile(r"^/c/en/([^/]+)")


def _conceptnet_term(uri: str) -> str | None:
    """Extract a readable English term from a ConceptNet concept URI."""
    match = _CN_EN_CONCEPT.match(uri)
    if not match:
        return None
    return match.group(1).replace("_", " ").strip()


def iter_conceptnet_facts(
    dump_path: str | Path,
    relations: Sequence[str] | None = None,
    min_weight: float = 1.0,
    max_facts: int | None = None,
    max_term_words: int = 5,
) -> Iterator[KnowledgeFact]:
    """Stream verbalised facts out of a ConceptNet assertions dump.

    The dump is a gzipped TSV with columns
    ``uri, relation, start, end, metadata-json`` and about 34M rows, so this is
    written as a streaming generator: memory stays flat regardless of dump size.

    Args:
        dump_path: Path to ``conceptnet-assertions-5.7.0.csv.gz`` (or the
            decompressed ``.csv``).
        relations: Relation whitelist; defaults to :data:`CONCEPTNET_RELATIONS`.
        min_weight: Drop edges below this ConceptNet weight. 1.0 keeps
            crowd-verified and high-agreement edges and removes most noise.
        max_facts: Stop after this many facts (useful for a bounded index).
        max_term_words: Drop edges whose concepts are longer than this; long
            concepts in ConceptNet are usually parse artefacts.

    Yields:
        :class:`KnowledgeFact` objects in dump order.
    """
    path = resolve(dump_path)
    templates = {
        name: CONCEPTNET_RELATIONS[name]
        for name in (relations or CONCEPTNET_RELATIONS)
        if name in CONCEPTNET_RELATIONS
    }
    if not templates:
        raise ValueError("No known ConceptNet relations selected")

    opener = gzip.open if path.suffix == ".gz" else open
    emitted = 0
    seen: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line_no, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _, relation_uri, start_uri, end_uri, meta_raw = parts[:5]
            relation = relation_uri.rsplit("/", 1)[-1]
            template = templates.get(relation)
            if template is None:
                continue
            head = _conceptnet_term(start_uri)
            tail = _conceptnet_term(end_uri)
            if not head or not tail or head == tail:
                continue
            if len(head.split()) > max_term_words or len(tail.split()) > max_term_words:
                continue
            try:
                weight = float(json.loads(meta_raw).get("weight", 1.0))
            except (json.JSONDecodeError, AttributeError, TypeError):
                weight = 1.0
            if weight < min_weight:
                continue
            text = template.format(head=head, tail=tail)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            yield KnowledgeFact(
                id=f"cn-{line_no}",
                text=text,
                source="ConceptNet",
                relation=relation,
                head=head,
                tail=tail,
                weight=weight,
            )
            emitted += 1
            if max_facts and emitted >= max_facts:
                return


# ---------------------------------------------------------------------- ATOMIC

#: ATOMIC / ATOMIC-2020 inferential relations and their verbalisations.
ATOMIC_RELATIONS: dict[str, str] = {
    "xIntent": "{head}, because PersonX wanted {tail}",
    "xNeed": "before {head}, PersonX needed {tail}",
    "xAttr": "{head}, so PersonX is seen as {tail}",
    "xEffect": "as a result of {head}, PersonX {tail}",
    "xReact": "after {head}, PersonX feels {tail}",
    "xWant": "after {head}, PersonX wants {tail}",
    "oEffect": "as a result of {head}, others {tail}",
    "oReact": "after {head}, others feel {tail}",
    "oWant": "after {head}, others want {tail}",
    "HinderedBy": "{head} can be hindered by {tail}",
    "isAfter": "{head} happens after {tail}",
    "isBefore": "{head} happens before {tail}",
    "HasSubEvent": "{tail} happens while {head}",
    "Causes": "{head} causes {tail}",
    "xReason": "{head} because {tail}",
}


def iter_atomic_facts(
    tsv_path: str | Path,
    relations: Sequence[str] | None = None,
    max_facts: int | None = None,
) -> Iterator[KnowledgeFact]:
    """Stream verbalised facts out of an ATOMIC / ATOMIC-2020 TSV.

    Expected format: three tab-separated columns ``head <TAB> relation <TAB>
    tail`` (the released ``train.tsv`` / ``dev.tsv`` / ``test.tsv`` layout).
    Rows whose tail is the ATOMIC "no annotation" marker ``none`` are skipped.
    """
    path = resolve(tsv_path)
    templates = {
        name: ATOMIC_RELATIONS[name]
        for name in (relations or ATOMIC_RELATIONS)
        if name in ATOMIC_RELATIONS
    }
    emitted = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head, relation, tail = parts[0].strip(), parts[1].strip(), parts[2].strip()
            template = templates.get(relation)
            if template is None or not head or not tail or tail.lower() == "none":
                continue
            head = head.replace("___", "something").strip()
            text = template.format(head=head, tail=tail)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            yield KnowledgeFact(
                id=f"atomic-{line_no}",
                text=text,
                source="ATOMIC",
                relation=relation,
                head=head,
                tail=tail,
                weight=1.0,
            )
            emitted += 1
            if max_facts and emitted >= max_facts:
                return


# --------------------------------------------------------------------- WordNet


def iter_wordnet_facts(
    nltk_data_dir: str | Path | None = None,
    max_synsets: int | None = None,
    include_sentiment: bool = True,
    min_sentiment: float = 0.25,
) -> Iterator[KnowledgeFact]:
    """Stream verbalised facts derived from WordNet 3.0 (+ SentiWordNet).

    Four fact families are emitted, all of which exist in WordNet -- nothing is
    authored by hand:

    * **gloss**      -- ``"<lemma> is defined as <gloss>"``
    * **hypernym**   -- ``"<lemma> is a type of <hypernym lemma>"``
    * **antonym**    -- ``"<lemma> is the opposite of <antonym lemma>"``
    * **sentiment**  -- ``"<lemma> carries a negative connotation"`` when
      SentiWordNet gives the synset a clear polarity. Evaluative polarity is the
      single most useful signal for spotting the positive-wording /
      negative-situation incongruity that characterises sarcasm.

    Args:
        nltk_data_dir: Directory containing the downloaded ``corpora/wordnet``.
            Added to ``nltk.data.path`` so the project can ship its own copy.
        max_synsets: Cap for quick smoke-test indices.
        include_sentiment: Emit SentiWordNet polarity facts.
        min_sentiment: Minimum |pos - neg| score for a polarity fact.

    Raises:
        RuntimeError: if the WordNet corpus is not available to NLTK.
    """
    import nltk

    if nltk_data_dir is not None:
        directory = str(resolve(nltk_data_dir))
        if directory not in nltk.data.path:
            nltk.data.path.insert(0, directory)

    try:
        from nltk.corpus import wordnet as wn

        wn.ensure_loaded()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "WordNet is not available to NLTK. Run:\n"
            "  python scripts/build_knowledge_index.py --download-wordnet\n"
            f"(underlying error: {exc})"
        ) from exc

    swn = None
    if include_sentiment:
        try:
            from nltk.corpus import sentiwordnet as _swn

            _swn.ensure_loaded()
            swn = _swn
        except Exception:  # noqa: BLE001 - sentiment facts are optional
            logger.warning("SentiWordNet unavailable; skipping polarity facts.")

    seen: set[str] = set()
    for index, synset in enumerate(wn.all_synsets()):
        if max_synsets is not None and index >= max_synsets:
            break
        lemmas = synset.lemmas()
        if not lemmas:
            continue
        name = lemmas[0].name().replace("_", " ")
        pos = synset.pos()

        gloss = (synset.definition() or "").strip()
        if gloss:
            yield from _emit(
                seen,
                KnowledgeFact(
                    id=f"wn-def-{synset.name()}",
                    text=f"{name} is defined as {gloss}",
                    source="WordNet",
                    relation="DefinedAs",
                    head=name,
                    tail=gloss,
                    metadata={"synset": synset.name(), "pos": pos},
                ),
            )

        for hypernym in synset.hypernyms()[:2]:
            h_lemmas = hypernym.lemmas()
            if not h_lemmas:
                continue
            h_name = h_lemmas[0].name().replace("_", " ")
            yield from _emit(
                seen,
                KnowledgeFact(
                    id=f"wn-isa-{synset.name()}-{hypernym.name()}",
                    text=f"{name} is a type of {h_name}",
                    source="WordNet",
                    relation="IsA",
                    head=name,
                    tail=h_name,
                    metadata={"synset": synset.name(), "pos": pos},
                ),
            )

        for lemma in lemmas:
            for antonym in lemma.antonyms():
                a_name = antonym.name().replace("_", " ")
                l_name = lemma.name().replace("_", " ")
                yield from _emit(
                    seen,
                    KnowledgeFact(
                        id=f"wn-ant-{synset.name()}-{antonym.name()}",
                        text=f"{l_name} is the opposite of {a_name}",
                        source="WordNet",
                        relation="Antonym",
                        head=l_name,
                        tail=a_name,
                        metadata={"synset": synset.name(), "pos": pos},
                    ),
                )

        if swn is not None:
            try:
                scores = swn.senti_synset(synset.name())
            except Exception:  # noqa: BLE001
                scores = None
            if scores is not None:
                polarity = scores.pos_score() - scores.neg_score()
                if abs(polarity) >= min_sentiment:
                    tone = "positive" if polarity > 0 else "negative"
                    yield from _emit(
                        seen,
                        KnowledgeFact(
                            id=f"wn-senti-{synset.name()}",
                            text=(
                                f"{name} ({gloss[:60]}) carries a {tone} connotation"
                                if gloss
                                else f"{name} carries a {tone} connotation"
                            ),
                            source="WordNet",
                            relation="HasSentiment",
                            head=name,
                            tail=tone,
                            weight=abs(polarity),
                            metadata={
                                "synset": synset.name(),
                                "pos": pos,
                                "pos_score": scores.pos_score(),
                                "neg_score": scores.neg_score(),
                            },
                        ),
                    )


def _emit(seen: set[str], fact: KnowledgeFact) -> Iterator[KnowledgeFact]:
    """Yield ``fact`` unless an identical verbalisation was already emitted."""
    key = fact.text.lower()
    if key in seen:
        return
    seen.add(key)
    yield fact


# ------------------------------------------------------------------ persistence


def write_facts(path: str | Path, facts: Iterable[KnowledgeFact]) -> int:
    """Write facts to JSONL; returns the number written."""
    from src.utils.io import write_jsonl

    return write_jsonl(path, (fact.to_dict() for fact in facts))


def read_facts(path: str | Path) -> list[KnowledgeFact]:
    """Read a JSONL knowledge file back into :class:`KnowledgeFact` objects."""
    from src.utils.io import read_jsonl

    return [KnowledgeFact.from_dict(row) for row in read_jsonl(path)]
