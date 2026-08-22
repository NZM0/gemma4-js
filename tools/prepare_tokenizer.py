from pathlib import Path
import argparse
import json
import shutil


FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the dependency-free Gemma 4 tokenizer files "
            "for the browser runtime."
        )
    )

    parser.add_argument(
        "model_dir",
        type=Path,
    )

    args = parser.parse_args()

    model_dir = (
        args.model_dir
        .expanduser()
        .resolve()
    )

    if not model_dir.exists():
        raise SystemExit(
            f"Model directory not found: {model_dir}"
        )

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    output_dir = (
        project_root
        /
        "tokenizer"
        /
        "gemma4"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    for name in FILES:
        source = (
            model_dir
            /
            name
        )

        if source.exists():
            shutil.copy2(
                source,
                output_dir / name,
            )

            copied.append(
                name
            )

    required = [
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing = [
        name
        for name in required
        if not (
            output_dir
            /
            name
        ).exists()
    ]

    if missing:
        raise SystemExit(
            "Missing required files: "
            +
            ", ".join(
                missing
            )
        )

    tokenizer_json = json.loads(
        (
            output_dir
            /
            "tokenizer.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    model = (
        tokenizer_json
        .get(
            "model",
            {}
        )
    )

    print(
        "Standalone tokenizer prepared."
    )

    print(
        "source:",
        model_dir
    )

    print(
        "target:",
        output_dir
    )

    print(
        "copied:",
        ", ".join(
            copied
        )
    )

    print(
        "tokenizer model:",
        model.get(
            "type"
        )
    )

    print(
        "vocab size:",
        len(
            model.get(
                "vocab",
                {}
            )
        )
    )

    print(
        "merges:",
        len(
            model.get(
                "merges",
                []
            )
        )
    )

    print(
        "byte_fallback:",
        model.get(
            "byte_fallback"
        )
    )

    print(
        "pre_tokenizer:",
        json.dumps(
            tokenizer_json.get(
                "pre_tokenizer"
            ),
            ensure_ascii=False,
        )[:1000]
    )

    print(
        "decoder:",
        json.dumps(
            tokenizer_json.get(
                "decoder"
            ),
            ensure_ascii=False,
        )[:1000]
    )


if __name__ == "__main__":
    main()
