test: lint
    uv run pytest tests/ --cov birdjepa --cov-report term-missing
    cd rust && cargo test

fmt:
    uvx ruff format --preview .
    cd rust && cargo fmt --all

lint: fmt
    uvx ruff check --fix .

# Type check (note: ty is prerelease, may have false positives)
types: fmt
    uvx ty check src/birdjepa
