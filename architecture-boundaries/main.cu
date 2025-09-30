/* Some of the things implemented here may seem simple, pointless, or otherwise
 * redundant (especially considering the information you can get directly from
 * cuda device properties). However, crafting kernels that stress, emphasize, or
 * reveal particular architecture properties is a useful way to test your
 * understanding of the architecture (and compiler, really).
 */

#include "architecture-boundaries.cuh"

int main() {
  //test_sm_count_driver();
  //test_warps_per_sm_driver();
  test_partitions_per_sm_driver();

  return 0;
}
