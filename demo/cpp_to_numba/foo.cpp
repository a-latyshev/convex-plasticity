#include "foo.hpp"

int foo(int a) {
  std::cout << "in foo(" << std::to_string(a) << ")" << std::endl;
  return a + 123;
}

namespace ns { namespace ns2 {
    double bar(double a) {
      std::cout << "in ns::ns2::bar(" << std::to_string(a) << ")" << std::endl;
      return a + 12.3;
    }
}}
