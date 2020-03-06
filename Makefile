init:
	pipenv install --dev
test:
	pipenv run python -m pytest
fmt:
	black .
	isort
