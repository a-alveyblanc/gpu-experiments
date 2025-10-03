#include <string>

#include "drivers.cuh"

int main(int argc, char **argv) {

  int n = 4096;

  for (int i = 1; i < argc; ++i) {
    std::string kernel = std::string(argv[i]);
    if (kernel == "--mod")
      drive_mod_indexing(n);
    else if (kernel == "--normal")
      drive_normal_indexing(n);
    else if (kernel == "--fast")
      drive_fast_indexing(n);
    else if (kernel == "--shared")
      drive_shared_memory_tiling(n);
    else if (kernel == "--shared-no-conflicts")
      drive_shared_memory_tiling_no_bank_conflicts(n);
    else if (kernel == "--threadblock_1d")
      drive_thread_block_1d_tiling(n);
    else if (kernel == "--threadblock_2d")
      drive_thread_block_2d_tiling(n);
    else if (kernel == "--all") {
      drive_normal_indexing(n);
      drive_mod_indexing(n);
      drive_fast_indexing(n);
      drive_shared_memory_tiling(n);
      drive_shared_memory_tiling_no_bank_conflicts(n);
      drive_thread_block_1d_tiling(n);
      drive_thread_block_2d_tiling(n);
    }
  }

  return 0;
}
