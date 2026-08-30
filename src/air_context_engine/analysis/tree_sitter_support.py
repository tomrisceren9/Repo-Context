"""Resmî Tree-sitter grammar paketleri için merkezi parser registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from tree_sitter import Language, Node, Parser


@dataclass(frozen=True)
class GrammarSpec:
    module: str
    factory: str = "language"


@dataclass(frozen=True)
class SymbolRange:
    signature: str
    name: str
    symbol_type: str
    start_line: int
    end_line: int


GRAMMAR_REGISTRY: dict[str, GrammarSpec] = {
    "c": GrammarSpec("tree_sitter_c"),
    "cpp": GrammarSpec("tree_sitter_cpp"),
    "c_sharp": GrammarSpec("tree_sitter_c_sharp"),
    "go": GrammarSpec("tree_sitter_go"),
    "java": GrammarSpec("tree_sitter_java"),
    "javascript": GrammarSpec("tree_sitter_javascript"),
    "python": GrammarSpec("tree_sitter_python"),
    "rust": GrammarSpec("tree_sitter_rust"),
    "typescript": GrammarSpec("tree_sitter_typescript", "language_typescript"),
    "tsx": GrammarSpec("tree_sitter_typescript", "language_tsx"),
}

EXTENSION_LANGUAGE: dict[str, str] = {
    ".c": "c",
    ".cc": "cpp",
    ".cjs": "javascript",
    ".cpp": "cpp",
    ".cs": "c_sharp",
    ".cts": "typescript",
    ".cxx": "cpp",
    ".go": "go",
    ".h": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".mts": "typescript",
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
}

SYMBOL_NODE_TYPES: dict[str, frozenset[str]] = {
    "c": frozenset(
        {
            "function_definition",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
        }
    ),
    "cpp": frozenset(
        {
            "field_declaration",
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "enum_specifier",
        }
    ),
    "c_sharp": frozenset(
        {
            "class_declaration",
            "constructor_declaration",
            "enum_declaration",
            "interface_declaration",
            "method_declaration",
            "record_declaration",
            "struct_declaration",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration", "type_spec"}),
    "java": frozenset(
        {
            "annotation_type_declaration",
            "class_declaration",
            "constructor_declaration",
            "enum_declaration",
            "interface_declaration",
            "method_declaration",
            "record_declaration",
        }
    ),
    "javascript": frozenset(
        {
            "class_declaration",
            "function_declaration",
            "generator_function_declaration",
            "lexical_declaration",
            "method_definition",
        }
    ),
    "rust": frozenset(
        {
            "enum_item",
            "function_item",
            "function_signature_item",
            "impl_item",
            "struct_item",
            "trait_item",
        }
    ),
    "typescript": frozenset(
        {
            "abstract_class_declaration",
            "abstract_method_signature",
            "class_declaration",
            "enum_declaration",
            "function_declaration",
            "generator_function_declaration",
            "interface_declaration",
            "lexical_declaration",
            "method_definition",
            "method_signature",
            "type_alias_declaration",
        }
    ),
    "tsx": frozenset(
        {
            "abstract_class_declaration",
            "abstract_method_signature",
            "class_declaration",
            "enum_declaration",
            "function_declaration",
            "generator_function_declaration",
            "interface_declaration",
            "lexical_declaration",
            "method_definition",
            "method_signature",
            "type_alias_declaration",
        }
    ),
}


def language_for_extension(extension: str) -> str | None:
    """Bir dosya uzantısını registry dil adına dönüştürür."""
    return EXTENSION_LANGUAGE.get(extension.lower())


def get_parser(language_name: str) -> Parser | None:
    """Desteklenen dil için parser döndürür; yükleme hatasında ``None`` verir."""
    spec = GRAMMAR_REGISTRY.get(language_name)
    if spec is None:
        return None
    try:
        module = import_module(spec.module)
        grammar_language: Callable[[], Any] = getattr(module, spec.factory)
        return Parser(Language(grammar_language()))
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


def _node_header(node: Node, source: bytes) -> str:
    """Bir bildirim node'unun gövdesiz, tek satırlı imza metnini döndürür."""
    raw = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    body = node.child_by_field_name("body")
    if body is not None:
        raw = source[node.start_byte : body.start_byte].decode(
            "utf-8", errors="replace"
        )
    elif "{" in raw:
        raw = raw.split("{", 1)[0]
    return " ".join(raw.rstrip(";{ \t\r\n").split())


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _symbol_name(node: Node, source: bytes, header: str) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, source)
    stack = list(node.children)
    while stack:
        child = stack.pop(0)
        if child.type in {
            "field_identifier",
            "identifier",
            "namespace_identifier",
            "type_identifier",
        }:
            return _node_text(child, source)
        stack[0:0] = list(child.children)
    match = __import__("re").search(r"([A-Za-z_]\w*)\s*(?:[<(]|$)", header)
    return match.group(1) if match else header


def _symbol_type(node_type: str) -> str:
    if "method" in node_type or node_type == "field_declaration":
        return "method"
    if "function" in node_type or node_type == "lexical_declaration":
        return "function"
    for category in (
        "class",
        "interface",
        "struct",
        "enum",
        "record",
        "namespace",
        "trait",
        "impl",
        "union",
        "type",
    ):
        if category in node_type:
            return category
    return "symbol"


def extract_symbol_ranges(language_name: str, code: str) -> list[SymbolRange]:
    """Registry diline göre sembol imzalarını ve kaynak aralıklarını çıkarır."""
    parser = get_parser(language_name)
    symbol_types = SYMBOL_NODE_TYPES.get(language_name)
    if parser is None or symbol_types is None:
        return []

    source = code.encode("utf-8")
    tree = parser.parse(source)
    symbols: list[SymbolRange] = []
    seen_ranges: set[tuple[int, int]] = set()

    def visit(node: Node, scope: tuple[str, ...] = ()) -> None:
        def has_descendant(node_type: str) -> bool:
            stack = list(node.children)
            while stack:
                child = stack.pop()
                if child.type == node_type:
                    return True
                stack.extend(child.children)
            return False

        is_cpp_method_declaration = (
            language_name == "cpp"
            and node.type == "field_declaration"
            and any(child.type == "function_declarator" for child in node.children)
        )
        is_arrow_declaration = node.type == "lexical_declaration" and has_descendant(
            "arrow_function"
        )
        is_symbol = node.type in symbol_types and (
            (node.type != "field_declaration" or is_cpp_method_declaration)
            and (node.type != "lexical_declaration" or is_arrow_declaration)
        )
        if is_symbol:
            byte_range = (node.start_byte, node.end_byte)
            if byte_range not in seen_ranges:
                seen_ranges.add(byte_range)
                header = _node_header(node, source)
                if header:
                    local_name = _symbol_name(node, source, header)
                    symbol_type = _symbol_type(node.type)
                    qualified_name = ".".join((*scope, local_name))
                    symbols.append(
                        SymbolRange(
                            signature=header,
                            name=qualified_name,
                            symbol_type=symbol_type,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                        )
                    )
                    if symbol_type in {
                        "class",
                        "interface",
                        "struct",
                        "namespace",
                        "trait",
                        "impl",
                        "record",
                    }:
                        scope = (*scope, local_name)
        for child in node.children:
            visit(child, scope)

    visit(tree.root_node)
    return symbols


def extract_symbol_signatures(language_name: str, code: str) -> list[str]:
    """Registry diline göre sınıf, tür, fonksiyon ve metot imzalarını çıkarır."""
    return [symbol.signature for symbol in extract_symbol_ranges(language_name, code)]
