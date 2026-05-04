from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


BYTE_VOCAB_SIZE = 256
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
LEADING_SPACE_KEY_PREFIX = "leading:"


@dataclass(frozen=True)
class TokenPiece:
    token_id: int
    text: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class TokenSegment:
    text: str
    start: int
    end: int
    kind: str
    pieces: tuple[TokenPiece, ...]

    @property
    def token_ids(self) -> list[int]:
        return [piece.token_id for piece in self.pieces]

    @property
    def piece_len(self) -> int:
        return len(self.pieces)


class HybridTokenizer:
    def __init__(
        self,
        char_tokens: list[str],
        surface_tokens: list[str],
        numeric_tokens: list[str],
        common_word_tokens: list[str],
        contextual_tokens: dict[str, str],
    ):
        self.special_tokens = [PAD_TOKEN, EOS_TOKEN]
        self.pad_token_id = BYTE_VOCAB_SIZE
        self.eos_token_id = BYTE_VOCAB_SIZE + 1
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.leading_token_to_base: dict[int, int] = {}
        self.base_token_to_leading: dict[int, int] = {}

        entries: list[tuple[str, str]] = []
        entries.extend((f"char:{token}", token) for token in char_tokens)
        entries.extend((f"surface:{token}", token) for token in surface_tokens)
        entries.extend((f"number:{token}", token) for token in numeric_tokens)
        entries.extend((f"common_word:{token}", token) for token in common_word_tokens)
        entries.extend((f"context:{name}", value) for name, value in contextual_tokens.items())

        seen_keys = set()
        next_id = BYTE_VOCAB_SIZE + len(self.special_tokens)
        base_entry_keys: list[str] = []
        for key, value in entries:
            if key in seen_keys:
                raise ValueError(f"duplicate vocab key {key!r}")
            seen_keys.add(key)
            self.token_to_id[key] = next_id
            self.id_to_token[next_id] = value
            base_entry_keys.append(key)
            next_id += 1

        for base_id in range(BYTE_VOCAB_SIZE):
            key = f"{LEADING_SPACE_KEY_PREFIX}byte:{base_id}"
            self.token_to_id[key] = next_id
            self.id_to_token[next_id] = bytes([base_id]).decode("utf-8", errors="replace")
            self.leading_token_to_base[next_id] = base_id
            self.base_token_to_leading[base_id] = next_id
            next_id += 1

        for base_key in base_entry_keys:
            base_id = self.token_to_id[base_key]
            key = f"{LEADING_SPACE_KEY_PREFIX}{base_key}"
            self.token_to_id[key] = next_id
            self.id_to_token[next_id] = self.id_to_token[base_id]
            self.leading_token_to_base[next_id] = base_id
            self.base_token_to_leading[base_id] = next_id
            next_id += 1

        self.char_tokens = sorted(set(char_tokens), key=len, reverse=True)
        self.surface_tokens = sorted(set(surface_tokens), key=len, reverse=True)
        self.numeric_tokens = sorted(set(numeric_tokens), key=len, reverse=True)
        self.common_word_tokens = sorted(set(common_word_tokens), key=len, reverse=True)
        self.char_token_set = set(char_tokens)

    @classmethod
    def from_file(cls, path: str | Path) -> "HybridTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            char_tokens=payload.get("char_tokens", []),
            surface_tokens=payload.get("surface_tokens", []),
            numeric_tokens=payload.get("numeric_tokens", []),
            common_word_tokens=payload.get("common_word_tokens", []),
            contextual_tokens=payload.get("contextual_tokens", {}),
        )

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE + len(self.special_tokens) + len(self.token_to_id)

    def decode_token_value(self, token_id: int) -> str:
        if token_id == self.pad_token_id or token_id == self.eos_token_id:
            return ""
        if token_id in self.leading_token_to_base:
            return " " + self.decode_token_value(self.leading_token_to_base[token_id])
        if 0 <= token_id < BYTE_VOCAB_SIZE:
            return bytes([token_id]).decode("utf-8", errors="replace")
        if token_id not in self.id_to_token:
            raise ValueError(f"unknown token id {token_id}")
        return self.id_to_token[token_id]

    def is_leading_space_token(self, token_id: int) -> bool:
        return token_id in self.leading_token_to_base

    def leading_space_token_id(self, token_id: int) -> int:
        try:
            return self.base_token_to_leading[token_id]
        except KeyError as exc:
            raise ValueError(f"token id {token_id} has no leading-space variant") from exc

    def match_literal(self, text: str, pos: int, tokens: list[str]) -> str | None:
        for token in tokens:
            if text.startswith(token, pos):
                return token
        return None

    def match_standalone_word(self, text: str, pos: int) -> str | None:
        for token in self.common_word_tokens:
            end = pos + len(token)
            if not text.startswith(token, pos):
                continue
            left_ok = pos == 0 or not (text[pos - 1].isalnum() or text[pos - 1] == "_")
            right_ok = end == len(text) or not (text[end].isalnum() or text[end] == "_")
            if left_ok and right_ok:
                return token
        return None

    def dot_kind(self, text: str, pos: int) -> str:
        prev_is_digit = pos > 0 and text[pos - 1].isdigit()
        next_is_digit = pos + 1 < len(text) and text[pos + 1].isdigit()
        if prev_is_digit and not next_is_digit:
            return "DOT_NUMERIC"
        return "DOT_SENTENCE"

    def plain_segment(self, text: str, pos: int) -> tuple[str, str]:
        if text[pos] == "_":
            end = pos + 1
            while end < len(text) and text[end] == "_":
                end += 1
            return text[pos:end], "underscore"

        match = WORD_PATTERN.match(text, pos)
        if match is None:
            return text[pos], "char"

        value = match.group(0)
        if value.isdecimal():
            return text[pos], "digit"
        if re.fullmatch(r"\w+", value, re.UNICODE):
            underscore_idx = value.find("_")
            if underscore_idx > 0:
                return value[:underscore_idx], "word"
            if underscore_idx == 0:
                end = 1
                while end < len(value) and value[end] == "_":
                    end += 1
                return value[:end], "underscore"
            first_is_digit = value[0].isdecimal()
            end = 1
            while end < len(value) and value[end].isdecimal() == first_is_digit:
                end += 1
            if end < len(value):
                if first_is_digit:
                    return value[0], "digit"
                return value[:end], "word"
            return value, "word"
        return value, "punct"

    def piece_fallback(self, value: str, start: int, kind: str) -> tuple[TokenPiece, ...]:
        pieces: list[TokenPiece] = []
        for char_offset, char in enumerate(value):
            char_start = start + char_offset
            char_end = char_start + 1
            char_key = f"char:{char}"
            if char in self.char_token_set:
                pieces.append(TokenPiece(self.token_to_id[char_key], char, char_start, char_end, "char"))
                continue
            for byte in char.encode("utf-8"):
                pieces.append(TokenPiece(byte, char, char_start, char_end, kind))
        return tuple(pieces)

    def single_piece_segment(
        self,
        key: str,
        value: str,
        start: int,
        end: int,
        kind: str,
    ) -> TokenSegment:
        return TokenSegment(
            text=value,
            start=start,
            end=end,
            kind=kind,
            pieces=(TokenPiece(self.token_to_id[key], value, start, end, kind),),
        )

    def apply_leading_space(self, segment: TokenSegment) -> TokenSegment:
        pieces = list(segment.pieces)
        first = pieces[0]
        pieces[0] = TokenPiece(
            self.leading_space_token_id(first.token_id),
            first.text,
            first.start,
            first.end,
            first.kind,
        )
        return TokenSegment(segment.text, segment.start, segment.end, segment.kind, tuple(pieces))

    def space_segment(self, pos: int) -> TokenSegment:
        return TokenSegment(" ", pos, pos + 1, "space", self.piece_fallback(" ", pos, "space"))

    def encode_segments(self, text: str, add_eos: bool = False) -> list[TokenSegment]:
        segments: list[TokenSegment] = []
        pos = 0
        pending_leading_space = False

        def append_segment(segment: TokenSegment):
            nonlocal pending_leading_space
            if pending_leading_space:
                segment = self.apply_leading_space(segment)
                pending_leading_space = False
            segments.append(segment)

        while pos < len(text):
            if text[pos] == " ":
                end = pos + 1
                while end < len(text) and text[end] == " ":
                    end += 1
                if end == len(text):
                    segments.extend(self.space_segment(space_pos) for space_pos in range(pos, end))
                else:
                    segments.extend(self.space_segment(space_pos) for space_pos in range(pos, end - 1))
                    pending_leading_space = True
                pos = end
                continue

            surface = self.match_literal(text, pos, self.surface_tokens)
            if surface is not None:
                end = pos + len(surface)
                append_segment(self.single_piece_segment(f"surface:{surface}", surface, pos, end, "surface"))
                pos = end
                continue

            numeric = self.match_literal(text, pos, self.numeric_tokens)
            if numeric is not None:
                end = pos + len(numeric)
                left_ok = pos == 0 or not text[pos - 1].isdigit()
                right_ok = end == len(text) or not text[end].isdigit()
                if left_ok and right_ok:
                    append_segment(self.single_piece_segment(f"number:{numeric}", numeric, pos, end, "number"))
                    pos = end
                    continue

            common_word = self.match_standalone_word(text, pos)
            if common_word is not None:
                end = pos + len(common_word)
                segment = self.single_piece_segment(
                    f"common_word:{common_word}",
                    common_word,
                    pos,
                    end,
                    "word",
                )
                append_segment(segment)
                pos = end
                continue

            if text[pos] == ".":
                name = self.dot_kind(text, pos)
                append_segment(self.single_piece_segment(f"context:{name}", ".", pos, pos + 1, name.lower()))
                pos += 1
                continue

            if text[pos].isspace():
                value = text[pos]
                append_segment(TokenSegment(value, pos, pos + 1, "space", self.piece_fallback(value, pos, "space")))
                pos += 1
                continue

            value, kind = self.plain_segment(text, pos)
            end = pos + len(value)
            append_segment(TokenSegment(value, pos, end, kind, self.piece_fallback(value, pos, "piece")))
            pos = end

        if add_eos:
            eos = TokenPiece(self.eos_token_id, EOS_TOKEN, len(text), len(text), "eos")
            segments.append(TokenSegment(EOS_TOKEN, len(text), len(text), "eos", (eos,)))
        return segments

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        return [
            piece.token_id
            for segment in self.encode_segments(text, add_eos=add_eos)
            for piece in segment.pieces
        ]

    def decode(self, token_ids: list[int]) -> str:
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                output.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue
            if token_id == self.eos_token_id:
                break
            if token_id in self.leading_token_to_base:
                base_id = self.leading_token_to_base[token_id]
                flush_bytes()
                output.append(" ")
                if 0 <= base_id < BYTE_VOCAB_SIZE:
                    byte_buffer.append(base_id)
                else:
                    output.append(self.decode_token_value(base_id))
                continue
            if 0 <= token_id < BYTE_VOCAB_SIZE:
                byte_buffer.append(token_id)
                continue
            flush_bytes()
            output.append(self.decode_token_value(token_id))

        flush_bytes()
        return "".join(output)


def default_vocab_path() -> Path:
    return Path(__file__).resolve().parent / "hybrid_surface_vocab.json"
