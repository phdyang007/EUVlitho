#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#define _USE_MATH_DEFINES
using namespace std;

int main (int argc,char* argv[])
{
 char eol[512];
 char com;
 int ncut=1901;
 int ncuts=1749;

 ifstream ifactor("factor/factor0.csv");
 ifstream ifactorx("factor/factorx.csv");
 ifstream ifactory("factor/factory.csv");
 double fac0[ncut],facx[ncuts],facy[ncuts];
 for(int i=0;i<ncut-1;i++)
  ifactor>>fac0[i]>>com;
 ifactor>>fac0[ncut-1];
 for(int i=0;i<ncuts-1;i++)
  ifactorx>>facx[i]>>com;
 ifactorx>>facx[ncuts-1];
 for(int i=0;i<ncuts-1;i++)
  ifactory>>facy[i]>>com;
 ifactory>>facy[ncuts-1];

  ifstream ipredict0r("predict/re0predict.csv");
  ifstream ipredict0i("predict/im0predict.csv");
  ifstream ipredictxr("predict/rexpredict.csv");
  ifstream ipredictxi("predict/imxpredict.csv");
  ifstream ipredictyr("predict/reypredict.csv");
  ifstream ipredictyi("predict/imypredict.csv");

 double d0re[ncut],d0im[ncut],dxre[ncuts],dxim[ncuts],dyre[ncuts],dyim[ncuts];
 for(int n=0;n<ncut;n++)
 { 
  ipredict0r>>d0re[n];
  d0re[n]=d0re[n]/fac0[n];
  ipredict0r.getline(eol, sizeof(eol));
 }
 for(int n=0;n<ncut;n++)
 {
  ipredict0i>>d0im[n];
  d0im[n]=d0im[n]/fac0[n];
  ipredict0i.getline(eol, sizeof(eol));
 }
 for(int n=0;n<ncuts;n++)
 {
  ipredictxr>>dxre[n];
  dxre[n]=dxre[n]/facx[n];
  ipredictxr.getline(eol, sizeof(eol)); 
 }
 for(int n=0;n<ncuts;n++)
 {
  ipredictxi>>dxim[n];
  dxim[n]=dxim[n]/facx[n];
  ipredictxi.getline(eol, sizeof(eol));
 }
 for(int n=0;n<ncuts;n++)
 {
  ipredictyr>>dyre[n];
  dyre[n]=dyre[n]/facy[n];
  ipredictyr.getline(eol, sizeof(eol));
 }
 for(int n=0;n<ncuts;n++)
 {
  ipredictyi>>dyim[n];
  dyim[n]=dyim[n]/facy[n];
  ipredictyi.getline(eol, sizeof(eol));
 }

 ofstream nnpredict("nnpredict.csv");
 for(int i=0;i<ncut;i++)
   nnpredict<<d0re[i]<<com;
 nnpredict<<endl;
 for(int i=0;i<ncut;i++)
   nnpredict<<d0im[i]<<com;
 nnpredict<<endl;
 for(int i=0;i<ncuts;i++)
   nnpredict<<dxre[i]<<com;
 nnpredict<<endl;
 for(int i=0;i<ncuts;i++)
   nnpredict<<dxim[i]<<com;
 nnpredict<<endl;
 for(int i=0;i<ncuts;i++)
   nnpredict<<dyre[i]<<com;
 nnpredict<<endl;
 for(int i=0;i<ncuts;i++)
   nnpredict<<dyim[i]<<com;
 nnpredict<<endl;
 return 0;
}



