// DexRT/source/GitVersion.hpp declares a global (non-namespaced) GIT_HASH,
// used by DexRT/source/PostProcessingCore.hpp. DexRT is a pinned submodule,
// so rather than running its own GitVersion generation, just forward
// Mosscap's own (namespaced) GIT_HASH, which is always up to date.
namespace Mosscap {
    extern const char* GIT_HASH;
}

const char* GIT_HASH = Mosscap::GIT_HASH;
