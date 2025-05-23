.PHONY: all build install clean

all: build

# Build target: creates the vesyla binary using the provided script
build:
	./scripts/make_appimage.sh

# Default installation prefix
# (overwrite with 'make PREFIX=/custom/path install')
PREFIX ?= /usr/local

# Install target: installs the vesyla binary to $(PREFIX)/bin
install:
	@if [ ! -d $(PREFIX)/bin ]; then \
		sudo mkdir -p $(PREFIX)/bin; \
	fi
	@if [ ! -f ./vesyla ]; then \
		echo "Error: vesyla not found. Please run 'make build' first."; \
		exit 1; \
	fi
	install -m 755 ./vesyla $(PREFIX)/bin/vesyla

# Clean target: removes the vesyla binary and build directory
clean:
	rm -rf ./vesyla build/

