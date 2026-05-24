.PHONY: debug release runh runhmt runc run-debug-history run-debug-history-mt run-release-history run-release-history-mt run-debug-cuda run-release-cuda hst hmt cuda clean

debug:
	cmake --preset debug
	cmake --build --preset debug

release:
	cmake --preset release
	cmake --build --preset release

run-debug-history: debug
	./build/debug/mc_history $(CONFIG)

run-debug-history-mt: debug
	./build/debug/mc_history_mt $(CONFIG)

run-release-history: release
	./build/release/mc_history $(CONFIG)

run-release-history-mt: release
	./build/release/mc_history_mt $(CONFIG)

run-debug-cuda: debug
	./build/debug/mc_cuda $(CONFIG)

run-release-cuda: release
	./build/release/mc_cuda $(CONFIG)

runhst: run-release-history

runh: runhst

runhmt: run-release-history-mt

runcuda: run-release-cuda

runc: runcuda

hst:
	./build/release/mc_history $(CONFIG)

hmt:
	./build/release/mc_history_mt $(CONFIG)

cuda:
	./build/release/mc_cuda $(CONFIG)

clean:
	rm -rf build
