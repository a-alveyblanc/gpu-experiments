/* Some of the things implemented here may seem simple, pointless, or otherwise
 * redundant (especially considering the information you can get directly from
 * cuda device properties). However, crafting kernels that stress, emphasize, or
 * reveal particular architecture properties is a useful way to test your
 * understanding of the architecture (and compiler, really).
 */

#include <stdio.h>
#include <string>
#include <vector>
#include "microbenchmarks.cuh"

void run_benchmarks(std::vector<std::string> benchmarks, int niters) {
  if (benchmarks.size() == 1 && benchmarks[0] == "all") {
    test_sm_count_driver(niters);
    test_warps_per_sm_driver(niters);
    test_partitions_per_sm_driver(niters);
    test_force_bank_conflicts_driver(niters);
    test_no_bank_conflicts_driver(niters);
    test_multicast_no_conflicts_driver(niters);
  }

  for (const auto& bench : benchmarks) {
    if (bench == "sm_count")
      test_sm_count_driver(niters);
    else if (bench == "warps_per_sm")
      test_warps_per_sm_driver(niters);
    else if (bench == "partitions_per_sm")
      test_partitions_per_sm_driver(niters);
    else if (bench == "conflict_suite") {
      test_force_bank_conflicts_driver(niters);
      test_no_bank_conflicts_driver(niters);
      test_multicast_no_conflicts_driver(niters);
    }
  }
}

void usage() {
  fprintf(stderr, 
    "Usage: `./arch --bench [comma separated list of benchmarks] --iters [integer]\n"
    "Available benchmarks: "
    "sm_count, {warps, partitions}_per_sm, conflict_suite\n");
}

int main(int argc, char** argv) {

  int niters;
  bool niters_set = 0;
  std::vector<std::string> benchmarks;
  for (int i = 1; i < argc; ++i) {

    if (!strcmp("--iters", argv[i])) {
      i += 1;
      niters = atoi(argv[i]);    
      niters_set = 1;
    }
    else if (!strcmp("--bench", argv[i])) {
      i += 1;
      std::string benchStr = std::string(argv[i]);
      size_t pos = 0;
      std::string bench;
      if (benchStr.find(",") != std::string::npos) {
        while (pos = benchStr.find(",") != std::string::npos) {
          std::string benchmark = benchStr.substr(0, pos);
          benchmarks.push_back(benchmark);
          benchStr.erase(0, pos + 1);
        }
      }
      else {
        benchmarks.push_back(std::string(argv[i]));
      }
    }
    else if (!strcmp("--help", argv[i]) || !strcmp("--usage", argv[i])) {
      usage();
      exit(-1);
    }
  }

  if (!niters_set) niters = 100'000;
  if (!benchmarks.size()) {
    fprintf(stderr, "Did not find any benchmarks to run, defaulting to `all`\n");
    benchmarks.push_back(std::string("all"));
  }

  fprintf(stderr, "Number of iterations: %d\n", niters);
  fprintf(stderr, "Running microbenchmark set: \n");
  for (const auto& bench : benchmarks)
    fprintf(stderr, "\t%s", bench.c_str());
  fprintf(stderr, "\n");

  run_benchmarks(benchmarks, niters);

  return 0;
}
