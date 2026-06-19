
.PHONY: format
format:
	poetry run black elvef
	poetry run black script
	poetry run isort elvef --profile black
	poetry run isort script --profile black

.PHONY: lint
lint:
	poetry run pylint elvef
	poetry run pylint script

.PHONY: check_type
check_type:
	poetry run mypy elvef
	poetry run mypy script

