.PHONY: debug release runh run-debug-history run-release-history clean

debug:
	cmake --preset debug
	cmake --build --preset debug

release:
	cmake --preset release
	cmake --build --preset release

run-debug-history: debug
	./build/debug/mc_history

run-release-history: release
	./build/release/mc_history

runh: run-release-history

clean:
	rm -rf build
