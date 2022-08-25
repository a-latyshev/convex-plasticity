#include <iostream>
int foo(int a);

namespace ns {
    
  namespace ns2 {
    double bar(double x);
    class BarCls {
    public:
      BarCls(double a): a_(a) {}
      double get_a() { return a_; }
      static int fun() { return 54321; }
    private:
      double a_;
    };
  }
}