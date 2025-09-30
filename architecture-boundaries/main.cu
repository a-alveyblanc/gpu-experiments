#include "architecture-boundaries.cuh"

int main() {
  test_sm_count_driver();
  
  // TODO
  test_warps_per_sm_driver();

  return 0;
}
