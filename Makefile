SHELL=/bin/bash
PROJECT_NAME=oat
PROJECT_PATH=oat/
LINT_PATHS=${PROJECT_PATH} examples/ benchmark/ test setup.py

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

test:
	$(call check_install, pytest)
	pytest -s

lint:
	$(call check_install, isort)
	$(call check_install, pylint)
	isort --check --diff --project=${LINT_PATHS}
	pylint -j 8 --recursive=y ${LINT_PATHS}

format:
	$(call check_install, autoflake)
	autoflake --remove-all-unused-imports -i -r ${LINT_PATHS}
	$(call check_install, black)
	black ${LINT_PATHS}
	$(call check_install, isort)
	isort ${LINT_PATHS}

check-docstyle:
	$(call check_install, pydocstyle)
	pydocstyle ${PROJECT_PATH} --convention=google

checks: lint check-docstyle

clean:
	@-rm -rf build/ dist/ .eggs/ site/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	@-find . -name '*.pyc' -type f -exec rm -rf {} +
	@-find . -name '__pycache__' -exec rm -rf {} +

package: clean
	PRODUCTION_MODE=yes python setup.py bdist_wheel

publish: package
	twine upload dist/*

addlicense:
	addlicense -c "Garena Online Private Limited" .

.PHONY: format lint check-docstyle checks