#include<fstream>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "./VBF.h" //mask pattern specification

int main (int argc,char* argv[])
{
 ofstream ofs("mask.csv");
 int NDIVX =2048; //mask pattern size (nm)
 int NDIVY =NDIVX;
 int NDIVSQ=NDIVX*NDIVY;
 int* mask2d=new int[NDIVSQ];
 maskgen(mask2d,NDIVX,NDIVY);
 for(int i=0;i<NDIVX;i++)
 for(int j=0;j<NDIVY;j++)
   ofs<<mask2d[NDIVY*i+j]<<",";
 ofs<<endl;

 ofstream ofsmask("maskimage.csv");
 ofsmask<<"data,1"<<endl;
 ofsmask<<"memo1"<<endl;
 ofsmask<<"memo2"<<endl;
 for(int i=0;i<NDIVX;i++)
  ofsmask<<","<<i;
 ofsmask<<endl;
 for (int j = 0; j < NDIVY; j++)
 {
  ofsmask<<j;
  for (int i = 0; i < NDIVX; i++)
  {
   ofsmask<<","<<mask2d[NDIVY*i + j];
  }
  ofsmask<<endl;
 }
 return 0;
}

