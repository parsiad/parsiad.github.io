.PHONY: all clean

PY_CONTENT := $(filter-out _%.py,$(wildcard *.py))  # Jupytext Python files (do not start with _ in the filename)
PY_PRIVATE := $(wildcard _*.py)  # Other Python files (start with _ in the filename)
TMP_DIR := ../tmp
NB_FILES := $(patsubst %.py,$(TMP_DIR)/%.ipynb,$(PY_CONTENT))
MD_DIR := ../content/blog
MD_FILES := $(patsubst %.py,$(MD_DIR)/%.md,$(PY_CONTENT))

all: $(MD_FILES)

$(TMP_DIR):
	mkdir -p $@

# Convert .py → .ipynb and execute in-place
# Execution is needed before conversion to Markdown otherwise TagRemovePreprocessor (see below) will not work
$(TMP_DIR)/%.ipynb: %.py $(PY_PRIVATE) | $(TMP_DIR)
	jupytext $< --to notebook --output $@
	PYTHONPATH=$(shell pwd) jupyter nbconvert $@ --execute --inplace --to notebook

# Convert executed notebook → .md
$(MD_DIR)/%.md: $(TMP_DIR)/%.ipynb
	jupyter nbconvert $< --output $(TMP_DIR)/$* --to markdown \
		--TagRemovePreprocessor.remove_cell_tags=no_cell \
		--TagRemovePreprocessor.remove_input_tags=no_input \
		--TagRemovePreprocessor.remove_all_outputs_tags=no_output
	mv $(TMP_DIR)/$*.md $(MD_DIR)/$*.md

clean:
	rm -f $(NB_FILES) $(MD_FILES)
	rmdir $(TMP_DIR)
