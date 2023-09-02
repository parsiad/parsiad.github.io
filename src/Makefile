ASSETS_DIR:=../assets
OUTPUT_DIR:=../_posts

IPYNB_FILES:=$(wildcard *.ipynb)
BASENAMES:=$(basename $(IPYNB_FILES))
CONTENT_DIRS:=$(addprefix $(ASSETS_DIR)/,$(addsuffix _files,$(BASENAMES)))
MD_FILES:=$(addprefix $(OUTPUT_DIR)/,$(addsuffix .md,$(BASENAMES)))

all: $(MD_FILES) $(CONTENT_DIRS)

$(OUTPUT_DIR)/%.md $(ASSETS_DIR)/%_files &: %.ipynb
	rm -fr $(OUTPUT_DIR)/$(basename $<).md $(ASSETS_DIR)/$(basename $<)_files
	mkdir -p $(OUTPUT_DIR)/$(basename $<)_files
	jupyter nbconvert --to markdown --output-dir=$(OUTPUT_DIR) --TagRemovePreprocessor.remove_cell_tags no_cell --TagRemovePreprocessor.remove_input_tags no_input $<
	# Replace instances of ![png]( with ![png](/assets/
	sed -i 's/^!\[png\](/!\[png\](\/assets\//g' $(OUTPUT_DIR)/$(basename $<).md
	# Replace { with \\{
	# sed -i 's/\\{/\\\\{/g' $(OUTPUT_DIR)/$(basename $<).md
	# Replace } with \\}
	# sed -i 's/\\}/\\\\}/g' $(OUTPUT_DIR)/$(basename $<).md
	mv $(OUTPUT_DIR)/$(basename $<)_files $(ASSETS_DIR)

clean:
	rm -fr $(CONTENT_DIRS) $(MD_FILES)

.PHONY: clean
