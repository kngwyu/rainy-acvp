ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(ARGS):;@:)

init:
	pipenv install --dev
test:
	pipenv run python -m pytest $(ARGS)
fmt:
	black .
	isort
