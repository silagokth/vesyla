.PHONY: build

build:
	./scripts/make_appimage.sh
	./scripts/make_appimage.sh

install:
	chmod +x ./vesyla
	sudo mv ./vesyla /usr/local/bin/vesyla

clean:
	rm -f ./vesyla
