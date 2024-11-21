#include<fstream>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "./VDF.h"

int main (int argc,char* argv[])
{
 ofstream ofs("mask.csv");
 int NDIVX =1024;
 int NDIVY =NDIVX;
 int NDIVSQ=NDIVX*NDIVY;
 int* mask2d=new int[NDIVSQ];
 maskgen(mask2d,NDIVX,NDIVY);
 for(int i=0;i<NDIVX;i++)
 for(int j=0;j<NDIVY;j++)
   ofs<<mask2d[NDIVY*i+j]<<",";
 ofs<<endl;
 return 0;
}

