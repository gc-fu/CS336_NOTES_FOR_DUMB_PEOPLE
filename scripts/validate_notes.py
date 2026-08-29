#!/usr/bin/env python3
"""Mechanical checks for the CS336 Markdown notes.

This script intentionally checks only properties a machine can verify.  A passing
result never replaces the Beginner Reviewer learning the note from beginning to
end and explicitly reporting that no understanding blockers remain.
"""

from __future__ import annotations

import re
import string
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"

REQUIRED_SECTIONS = {
    "复习卡": r"^##+ .*复习卡",
    "前置知识": r"^##+ .*前置知识",
    "常见误区": r"^##+ .*常见误区",
    "自测题": r"^##+ .*自测",
    "答案": r"^##+ .*答案",
    "视频导航": r"^##+ .*视频.*导航",
    "来源": r"^##+ .*来源",
}

PLACEHOLDERS = re.compile(r"\b(?:TODO|TBD|FIXME)\b|待补|待完善", re.IGNORECASE)
GITHUB_FORBIDDEN_MATH_MACROS = {
    "operatorname": r"\mathrm{...}",
}
GITHUB_UNSUPPORTED_MATH_DELIMITERS = {
    r"\\\(": r"\(...\)（请改用 $`...`$）",
    r"^[ \t]*\\\[[ \t]*$": r"\[...\]（请改用顶格的 ```math 围栏）",
}
GITHUB_FRAGILE_INLINE_MATH = re.compile(r"(?<![\\$`])\$(?![$`])")
GITHUB_INDENTED_MATH_FENCE = re.compile(r"^[ \t]+```math[ \t]*$", re.MULTILINE)
ASTERISK_RUN = re.compile(r"(?<!\\)\*{2,}")
UNESCAPED_TABLE_PIPE = re.compile(r"(?<!\\)\|")


def mask_markdown_code(text: str) -> str:
    """Replace code with spaces while preserving offsets and line numbers."""

    def replace_chars(match: re.Match[str], fill: str) -> str:
        return re.sub(r"[^\r\n]", fill, match.group(0))

    masked = re.sub(
        r"```.*?```",
        lambda match: replace_chars(match, " "),
        text,
        flags=re.DOTALL,
    )
    return re.sub(
        r"(`+).*?\1",
        lambda match: replace_chars(match, "x"),
        masked,
    )


def is_github_punctuation(character: str) -> bool:
    """Match the punctuation definition used by GitHub's CommonMark parser."""

    return character in string.punctuation or unicodedata.category(character).startswith(
        "P"
    )


def find_unspaced_strong(text: str) -> int | None:
    """Return the offset of a strong delimiter that GitHub cannot disambiguate."""

    masked = mask_markdown_code(text)
    offset = 0
    for raw_line, masked_line in zip(
        text.splitlines(keepends=True), masked.splitlines(keepends=True)
    ):
        delimiters: list[int] = []
        for match in ASTERISK_RUN.finditer(masked_line):
            if len(match.group(0)) == 4 and masked_line.strip() != match.group(0):
                return offset + match.start() + 2
            delimiters.extend(
                match.start() + 2 * index
                for index in range(len(match.group(0)) // 2)
            )
        for closing in delimiters[1::2]:
            following = closing + 2
            if following >= len(raw_line) or raw_line[following].isspace():
                continue
            preceding_character = raw_line[closing - 1]
            following_character = raw_line[following]
            if is_github_punctuation(
                preceding_character
            ) and not is_github_punctuation(following_character):
                return offset + following
        offset += len(raw_line)
    return None


def check_note(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors: list[str] = []

    # Markdown-looking lines inside examples (for example Python comments that
    # start with "# ") are not document headings.
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    if len(lines) < 500:
        errors.append(f"只有 {len(lines)} 行，低于完整讲义的机械下限 500 行")
    if len(text) < 20_000:
        errors.append(f"只有 {len(text)} 个字符，低于机械下限 20,000 字符")

    h1_count = len(re.findall(r"^# (?!#)", prose, flags=re.MULTILINE))
    if h1_count != 1:
        errors.append(f"一级标题应恰好 1 个，实际 {h1_count} 个")

    if text.count("```") % 2:
        errors.append("代码围栏 ``` 数量为奇数")
    if text.count("$$") % 2:
        errors.append("块公式分隔符 $$ 数量为奇数")

    match = PLACEHOLDERS.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        errors.append(f"第 {line} 行存在占位内容：{match.group(0)!r}")

    for macro, replacement in GITHUB_FORBIDDEN_MATH_MACROS.items():
        match = re.search(rf"\\{re.escape(macro)}\b", prose)
        if match:
            errors.append(
                f"使用 GitHub 禁止的公式宏 \\{macro}；请改用 {replacement}"
            )

    for pattern, delimiter in GITHUB_UNSUPPORTED_MATH_DELIMITERS.items():
        if re.search(pattern, prose, flags=re.MULTILINE):
            errors.append(f"使用 GitHub 不支持的公式分隔符 {delimiter}")

    prose_without_inline_code = re.sub(r"(`+).*?\1", "", prose)
    if GITHUB_FRAGILE_INLINE_MATH.search(prose_without_inline_code):
        errors.append(
            "使用易受 Markdown 冲突影响的行内公式 $...$；"
            "请改用 GitHub 推荐的 $`...`$"
        )

    if GITHUB_INDENTED_MATH_FENCE.search(text):
        errors.append(
            "使用 GitHub 可能按普通代码显示的缩进 math 围栏；"
            "列表内请改用单独一行的 $`...`$"
        )

    unspaced_strong = find_unspaced_strong(text)
    if unspaced_strong is not None:
        line = text.count("\n", 0, unspaced_strong) + 1
        errors.append(
            f"第 {line} 行的粗体结束标记后缺少分隔空格；"
            "请在相邻文字或下一段粗体前添加空格，以便 GitHub 正确显示"
        )

    for line_number, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            continue
        for formula in re.findall(r"\$`(.*?)`\$", line):
            if UNESCAPED_TABLE_PIPE.search(formula):
                errors.append(
                    f"第 {line_number} 行的表格公式含未转义竖线；"
                    r"条件符请用 \mid，定界符请用 \lvert/\rvert"
                )

    for label, pattern in REQUIRED_SECTIONS.items():
        if not re.search(pattern, prose, flags=re.MULTILINE | re.IGNORECASE):
            errors.append(f"缺少章节：{label}")

    if "youtube.com/watch?v=" not in text:
        errors.append("缺少官方 YouTube 视频链接")
    if "stanford-cs336/lectures" not in text and "cs336.stanford.edu" not in text:
        errors.append("缺少 Stanford 官方课程材料链接")

    timestamp_links = re.findall(r"youtube\.com/watch\?v=[\w-]+(?:&amp;|&)t=\d+s", text)
    if len(timestamp_links) < 8:
        errors.append(f"可点击视频时间戳只有 {len(timestamp_links)} 个，至少应有 8 个")

    numbered_items = re.findall(r"^\d+\.\s+", prose, flags=re.MULTILINE)
    if len(numbered_items) < 30:
        errors.append(
            f"编号题目/答案项总计只有 {len(numbered_items)} 个；"
            "至少应支持 15 道题及其答案"
        )

    source_labels = sum(
        text.count(label)
        for label in ("【课程】", "【视频补充】", "【补充】", "【补充解释】", "【延伸】")
    )
    if source_labels < 5:
        errors.append(f"正文来源标签只有 {source_labels} 个，课程与补充边界可能不清楚")

    return errors


def main() -> int:
    # Windows terminals may default to cp1252 even though every note is UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    paths = sorted(NOTES.glob("lecture_*.md"))
    if not paths:
        print("ERROR: notes/ 下没有 lecture_*.md", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        errors = check_note(path)
        if errors:
            failed = True
            print(f"FAIL {path.relative_to(ROOT)}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {path.relative_to(ROOT)}")

    print("\n提醒：机械检查通过不等于 Beginner Reviewer 通过。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
