.PHONY: debug release runh runc run-debug-history run-release-history run-debug-cuda run-release-cuda h c clean

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

run-debug-cuda: debug
	./build/debug/mc_cuda

run-release-cuda: release
	./build/release/mc_cuda

runh: run-release-history

runc: run-release-cuda

h:
	./build/release/mc_history

c:
	./build/release/mc_cuda

clean:
	rm -rf build
