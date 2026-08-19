#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reference token-id streams from the checkpoint's own tokenizer.json.

WHY NOT THE SHIPPED encoding/tests
    They are CHAT-TEMPLATE tests: the inputs are message lists and the outputs
    are formatted TEXT ("<|begin_of_sentence|>You are a helpful assistant..."),
    not token ids. They validate encoding_dsv4.py's message formatting and say
    nothing about the BPE. This file gets ids from HF tokenizers instead, which
    is the same Rust implementation the checkpoint was built against.

WHAT THE CASES ARE FOR
    Random prose would pass on almost every way a byte-level BPE can be wrong.
    Each case below targets one part of the pre-tokenizer, because that is where
    the disagreements live -- the merge loop is mechanical once the splits are
    right.

      digits        \\p{N}{1,3} isolates runs of 1-3 digits, so 1234 does NOT
                    split the same way as 123 or 12345
      cjk           its own Split, and the ranges are specific: CJK ideographs
                    plus hiragana plus katakana, NOT "anything non-Latin"
      punct_letter  the third regex's FIRST alternative, punctuation followed
                    by letters, which must win over the later alternatives
      ws            \\s+(?!\\S) and \\s*[\\r\\n]+ differ from plain \\s+, and
                    trailing whitespace is the case that separates them
      marks         \\p{M} combining marks attach to the letter run
      mixed         everything at once, which is what real text looks like
"""
import json
import os
import pathlib
import sys

from tokenizers import Tokenizer

CASES = {
    "ascii":        "The capital of France is Paris.",
    "digits":       "1 12 123 1234 12345 007 3.14159 2026",
    "cjk":          "你好世界 こんにちは "
                    "カタカナ 漢字",
    "punct_letter": "!abc @def #ghi $jkl %mno ^pqr &stu *vwx (yz)",
    "punct_run":    "... !!! ?!?! --- ***  ,,, ;;; ::",
    "ws":           "a  b   c\td\n\ne\r\nf   \n   g   ",
    "marks":        "café naïve नमस्ते "
                    "كَتَبَ",
    "emoji":        "hello \U0001f600 world \U0001f1ec\U0001f1e7 flag "
                    "\U0001f469‍\U0001f4bb dev",
    "code":         "def f(n):\n    return [x**2 for x in range(n) if x % 2 == 0]\n",
    "mixed":        "Q3 2026: revenue 上昇 12.5% — see §4.2 "
                    "(p. 17). Привет! "
                    "\U0001f680 100%\n\ndone.",
    "leading_ws":   "   leading spaces and a tab\there",
    "empty":        "",
    "single":       "a",
    "newlines":     "\n\n\n\n",
}


def main():
    model = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                               else "~/models/dsv4-flash")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                       else "tests/fixtures/tok")
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))

    cases = {}
    for name, text in CASES.items():
        enc = tok.encode(text, add_special_tokens=False)
        cases[name] = {"text": text, "ids": enc.ids, "tokens": enc.tokens}
        print(f"  {name:<13} {len(text):>4} chars -> {len(enc.ids):>4} ids")

    body = {
        "tokenizers_version": __import__("tokenizers").__version__,
        "vocab_size": tok.get_vocab_size(),
        "cases": cases,
    }
    (out / "tok_cases.json").write_text(json.dumps(body) + "\n")
    total = sum(len(c["ids"]) for c in cases.values())
    print(f"\n{len(cases)} cases, {total} reference ids -> {out}/tok_cases.json")


if __name__ == "__main__":
    main()
