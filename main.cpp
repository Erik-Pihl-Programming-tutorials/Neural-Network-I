/********************************************************************************
* main.cpp: Implementering av ett enkelt neuralt nätverk i C++.
*           Under del 1 implementeras ett dense-lager.
********************************************************************************/
#include <vector>
#include "dense_layer.hpp"

/********************************************************************************
* main: Skapar ett dense-lager och tränar detta under tio epoker med en
*       lärhastighet på 10 %. Efter träningen genomförs utskrift av 
*       dense-lagrets parametrar.
********************************************************************************/
int main(void)
{
   const std::vector<double> train_in = { 1, 1, 1, 1 };
   const std::vector<double> train_out = { 2, 2, 2 };

   dense_layer d1(3, 4);

   for (auto i = 0; i < 100; ++i)
   {
      d1.feedforward(train_in);
      d1.backpropagate(train_out);
      d1.optimize(train_in, 0.1);
   }

   d1.print();

   return 0;
}