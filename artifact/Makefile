.PHONY: verify values audit test manifest

verify:
	python3 scripts/verify_results.py
	python3 scripts/verify_paper_values.py

values:
	python3 scripts/verify_paper_values.py

audit:
	python3 scripts/audit_anonymity.py

test:
	PYTHONPATH=.:src python3 -m pytest

manifest:
	python3 scripts/build_manifest.py

