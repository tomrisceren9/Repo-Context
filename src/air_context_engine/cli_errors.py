import json
from collections.abc import Sequence
from typing import Any, NoReturn

import click


class CLIError(click.ClickException):
    """Makine tarafından okunabilir, kararlı CLI hata sözleşmesi."""

    exit_code = 1

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        exit_code: int = 1,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.__dict__["exit_code"] = exit_code

    def format_message(self) -> str:
        return json.dumps(
            {
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "details": self.details,
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def show(self, file: Any | None = None) -> None:
        click.echo(self.format_message(), file=file or click.get_text_stream("stderr"))


class StructuredGroup(click.Group):
    """Click kullanım ve çalışma hatalarını aynı JSON sözleşmesine dönüştürür."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        if not standalone_mode:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )

        try:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )
        except CLIError as exc:
            exc.show()
            raise SystemExit(exc.exit_code) from exc
        except click.ClickException as exc:
            structured = CLIError(
                "CLI_USAGE_ERROR",
                exc.format_message(),
                details={"exception": type(exc).__name__},
                exit_code=exc.exit_code,
            )
            structured.show()
            raise SystemExit(structured.exit_code) from exc
        except click.Abort as exc:
            structured = CLIError(
                "CLI_ABORTED", "Komut kullanıcı tarafından iptal edildi."
            )
            structured.show()
            raise SystemExit(structured.exit_code) from exc


def fail(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    exit_code: int = 1,
) -> NoReturn:
    raise CLIError(code, message, details=details, exit_code=exit_code)
