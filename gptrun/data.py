from dataclasses import dataclass
from itertools import takewhile
import ast
import doctest
import textwrap


@dataclass
class InvokationExample:
    source: str
    call_args: list[str]
    call_args_obj: list[object]
    call_kwargs: dict[str, str]
    call_kwargs_obj: dict[str, object]
    want_obj: object
    want: str
    options: dict

    @classmethod
    def from_doctest_example(cls, example):
        source = ast.parse(example.source).body[0]
        if not isinstance(source.value, ast.Call):
            raise ValueError("Examples must be function calls only!")
        want = ast.unparse(ast.parse(example.want).body[0])
        call_args = [ast.unparse(x) for x in source.value.args]
        call_kwargs = {kw.arg: ast.unparse(kw.value) for kw in source.value.keywords}
        return cls(
            source=ast.unparse(source),
            call_args=call_args,
            call_args_obj=[ast.literal_eval(a) for a in call_args],
            call_kwargs=call_kwargs,
            call_kwargs_obj={k: ast.literal_eval(v) for k, v in call_kwargs.items()},
            want_obj=ast.literal_eval(want),
            want=want,
            options=example.options,
        )


@dataclass
class FakeFunctionDefinition:
    summary: str
    examples: list[InvokationExample]

    @staticmethod
    def get_summary(docstring):
        """Return the first paragraph of a function's docstring."""
        dedented_docstring = textwrap.dedent(docstring)
        paragraphs = dedented_docstring.strip().splitlines()
        prelude = takewhile(lambda p: not p.startswith(">>>"), paragraphs)
        return "\n".join(prelude)

    @classmethod
    def from_docstring(cls, docstring, external_examples=None):
        """Initialize a FakeFunctionDefinition from a docstring."""
        summary = cls.get_summary(docstring)
        if external_examples:
            examples = doctest.DocTestParser().get_examples(external_examples)
        else:
            examples = doctest.DocTestParser().get_examples(docstring)

        return cls(
            summary=summary,
            examples=[InvokationExample.from_doctest_example(e) for e in examples],
        )
