# ---------------------------------------------------------------------------
# Defaults — override on the command line, e.g. `make test PYTEST_ARGS=-v`
# ---------------------------------------------------------------------------
PYTHON        ?= python
COMPOSE       ?= docker compose
NETWORK       ?= mlops-net
PYTEST_ARGS   ?= -n auto
FLAKE8_ARGS   ?= --max-line-length=120
LINT_PATH     ?= src
TEST_PATH     ?= src/tests
CI_IMAGE      ?= diabetes-prediction-app

.PHONY: help up down build-feast app lint test \
	build-image push-image prepare train online-infer

help:
	@echo "Local (docker compose):"
	@echo "  make up           - bring up the infra (garage, mlflow, feast, ...)"
	@echo "  make down         - tear down the infra"
	@echo "  make build-feast  - build the feast server image"
	@echo "  make run          - run the app pipeline (diabetes-app, profile 'app')"
	@echo ""
	@echo "CI (pipeline stages):"
	@echo "  make lint         - flake8 over $(LINT_PATH)"
	@echo "  make test         - pytest over $(TEST_PATH)"
	@echo "  make build-image  - docker buildx build -t \$$(CI_IMAGE) ."
	@echo "  make push-image   - docker push \$$(CI_IMAGE)"
	@echo "  make prepare      - python -m src.main prepare"
	@echo "  make train        - python -m src.main train"
	@echo "  make online-infer - python -m src.main online-infer"

# ---------------------------------------------------------------------------
# Local: docker compose (see docker-compose.yml for service details)
# ---------------------------------------------------------------------------
up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

build-feast:
	$(COMPOSE) build feast

run:
	$(COMPOSE) --profile app up diabetes-app

# ---------------------------------------------------------------------------
# CI: test & lint (used by .gitlab/ci/test.gitlab-ci.yml)
# ---------------------------------------------------------------------------
lint:
	$(PYTHON) -m flake8 $(LINT_PATH) $(FLAKE8_ARGS)

test:
	$(PYTHON) -m pytest $(TEST_PATH) $(PYTEST_ARGS)

# ---------------------------------------------------------------------------
# CI: build/prepare/train/online-infer pipeline stages
# (used by .gitlab/ci/{build,prepare-data,train,online-infer}.gitlab-ci.yml)
# ---------------------------------------------------------------------------
build-image:
	docker buildx build -t $(CI_IMAGE) .

push-image:
	docker push $(CI_IMAGE)

prepare:
	$(PYTHON) -m src.main prepare

train:
	$(PYTHON) -m src.main train

online-infer:
	$(PYTHON) -m src.main online-infer
