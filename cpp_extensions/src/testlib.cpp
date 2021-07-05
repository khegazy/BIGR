//http://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html#id3
#include <iostream>
#include "testlib.h"

using namespace std;

void myprint() {
  cout << "Hello python world" << endl;
  }

float mult(float a, double b) {
  cout << "in cpp " << a*b << endl;
  return (float)(a*b);
}

void arrtest(double* a, int len) {
  cout<<"IN cpp: ";
  for (int i=0; i<len; i++) {
    cout<<a[i]<<"/";
    a[i] *= 3.1;
    cout<<a[i]<<"  ";
  }
  cout<<endl;
}

void tdarrtest(double** a, int len) {
  cout<<"IN cpp: ";
  for (int i=0; i<len; i++) {
    for (int j=0; j<len; j++) {
      cout<<a[i][j]<<"/";
      a[i][j] *= 3.1;
      cout<<a[i][j]<<"  ";
    }
  }
  cout<<endl;
}


