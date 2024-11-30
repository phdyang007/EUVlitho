#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <stdlib.h>
using namespace std;
#define MKL_NUM_THREADS 100
#include "mkl_dfti.h"
#define _USE_MATH_DEFINES
#include <omp.h>
#define EIGEN_USE_MKL_ALL
#include "Eigen/Eigen"
#include "magma_v2.h"
#include "../include/header.h"

#define _Complex_I (1.0iF) 
#define XDIV 512  //XDIV=NDIVX/MX
#define FDIV 2048 //FDIV=NDIVX
#define nout 64 // maximum frequency for FFT

int main (int argc,char* argv[])
{
 std::chrono::system_clock::time_point  start, now;
 double elapsed;
 start = std::chrono::system_clock::now(); 

 double pi=atan(1.)*4.;
 complex<double> zi (0., 1.);
 char eol[512];
 char com;

 ifstream imask("./mask.csv"); //input mask pattern
 ifstream ipredict("nnpredict.csv"); //input M3D parameters
 ofstream ofsint("nnsocs.csv"); //output image intensity by CNN

 int ncut=1901; //number of the (l,m) pairs for a0
 int ncuts=1749; //number of the (l,m) pairs for ax and ay

 int nsocs=100; //total kernel numbers for TCC
 int nsocsxy=20; //total kernel number for TCCx and TCCy

 int MX=4; //X magnification
 int MY=4; //Y magnification
 int NDIVX=2048; //X pitch (nm)
 int NDIVY=NDIVX; //Y pitch (nm)
 double dx =NDIVX;
 double dy =NDIVY;

 complex<double> nta(0.9567,0.0343); //absorber complex refractive index
 vector< complex<double>> eabs(100); //eabs[0]: top absorber layer dielectric constant, eabs[1]: 2nd absorber layer...
 vector<double> dabs(100); // dabs[0]: top absorber layer thickness, dabs[1]: 2nd absorber layer...
 int NML=40; //number of the multilayer pairs
 int NABS=1; //number of the absorber layers
 eabs[0]=nta*nta; 
 double dabst=60.; //total aborber thickness (nm)
 dabs[0]=dabst;
 double z=0; //defocus
 double z0=dabst+42.; //reflection point inside ML from the top of the absorber

 double lambda,theta0,azimuth,phi0;
 lambda = 13.5; //wavelength (nm)
 double k = 2.*pi/lambda;
 theta0 = -6.; //chief ray angle (degree)
 azimuth =0.; //azimuthal angle (degree)
 phi0 = 90. - azimuth;
 double NA,sigma1,sigma2,openangle;
 NA = 0.33;
 int type=2; //0: circular, 1: annular, 2: dipole
 sigma1 = 0.9; //outer sigma
 sigma2=0.55; //inner sigma
 openangle = 90.; //opening angle for dipole illumination

 double sx0, sy0;
 sx0 = k*sin(pi / 180.*theta0)*cos(pi / 180.*phi0);
 sy0 = k*sin(pi / 180.*theta0)*sin(pi / 180.*phi0);

 double delta=1.;
 int FDIVX=dx/delta+0.000001; 
 int FDIVY=dx/delta+0.000001; 

 int NDIVSQ=NDIVX*NDIVY;
 int* mask2d=new int[NDIVSQ];
 Eigen::MatrixXcd pattern(FDIVX,FDIVY);

 int lsmaxX=NA*dx/double(MX)/lambda+1;
 int lsmaxY=NA*dy/double(MY)/lambda+1;
 int lpmaxX=NA*dx/double(MX)*2/lambda+0.0001;
 int lpmaxY=NA*dy/double(MY)*2/lambda+0.0001;
 int nsourceX=2*lsmaxX+1;
 int nsourceY=2*lsmaxY+1;
 int noutX=2*lpmaxX+1;
 int noutY=2*lpmaxY+1;

 int FDIVX1 = FDIVX + 1;
 int FDIVY1 = FDIVY + 1;
 vector< complex<double> > cexpX(FDIVX1),cexpY(FDIVY1);
 exponential(cexpX, pi, FDIVX);
 exponential(cexpY, pi, FDIVY);

 double cutx,cuty;
 cutx = NA/MX*1.5;
 cuty = NA/MY*1.5;

 int LMAX = cutx*dx / lambda;
 int Lrange = 2 * LMAX + 1;
 int Lrange2 = 4 * LMAX + 1;
 int MMAX = cuty*dy / lambda;
 int Mrange = 2 * MMAX + 1;
 int Mrange2 = 4 * MMAX + 1;

 int Nrangep = 0;
 vector<int> lindexp, mindexp;
 for (int i = 0; i < Lrange; i++)
 {
  int ii = i - LMAX;
  for (int j = 0; j < Mrange; j++)
  {
   int jj = j - MMAX;
   if (pow(ii/ double(LMAX + 0.01), 2.)+pow(jj/ double(MMAX + 0.01), 2.) <= 1.)
   {
    lindexp.push_back(ii);
    mindexp.push_back(jj);
    Nrangep++;
   }
  }
 }

 vector<vector<Eigen::VectorXcd>> Axvc(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrangep)));
 for(int i=0;i<NDIVX;i++)
  {
   for(int j=0;j<NDIVY;j++)
    mask2d[NDIVY*i+j]=0;
  }
 ampS('X',Axvc, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrangep, lindexp, mindexp, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, sx0, sy0, eabs, dabs, cexpX,cexpY);

 vector<vector<Eigen::VectorXcd>> Axab(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrangep)));
 for(int i=0;i<NDIVX;i++)
  {
   for(int j=0;j<NDIVY;j++)
    mask2d[NDIVY*i+j]=1;
  }
 ampS('X',Axab, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrangep, lindexp, mindexp, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, sx0, sy0, eabs, dabs, cexpX,cexpY);

 cutx = NA/MX*6.;
 cuty = NA/MY*6.;
 LMAX = cutx*dx / lambda;
 Lrange = 2 * LMAX + 1;
 Lrange2 = 4 * LMAX + 1;
 MMAX = cuty*dy / lambda;
 Mrange = 2 * MMAX + 1;
 Mrange2 = 4 * MMAX + 1;

 int Nrange = 0;
 vector<int> lindex, mindex;
 for (int i = 0; i < Lrange; i++)
 {
  int ii = i - LMAX;
  for (int j = 0; j < Mrange; j++)
  {
   int jj = j - MMAX;
   if ((abs(ii) / double(LMAX + 0.01) + 1.)*(abs(jj) / double(MMAX + 0.01) + 1.) <= 2.)
//   if (pow(ii/ double(LMAX + 0.01), 2.)+pow(jj/ double(MMAX + 0.01), 2.) <= 1.)
   {
    lindex.push_back(ii);
    mindex.push_back(jj);
    Nrange++;
   }
  }
 }

 int nrange=Nrange;
 int ninput=0;
 vector<int> linput(nrange),minput(nrange),xinput(nrange);
 for(int ip=0;ip<noutX;ip++) 
 {
  for(int jp=0;jp<noutY;jp++) 
  {
   int snum=0;
   for(int is=0;is<nsourceX;is++)
   {
     for(int js=0;js<nsourceY;js++)
     {
      if(((pow((is-lsmaxX)*MX/dx,2)
           +pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2))
          &&((pow((ip-lpmaxX+is-lsmaxX)*MX/dx,2)
           +pow((jp-lpmaxY+js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)))
       {
        snum+=1;
       }
     }
    }
   if(snum>0)
   {
    linput[ninput]=ip-lpmaxX;
    minput[ninput]=jp-lpmaxY;
//    if(snum>=12)
    if(snum>=8)
      xinput[ninput]=1;
    else
      xinput[ninput]=0;           
    ninput++;
   }
  }
 }

 Eigen::MatrixXcd vcxx(nsourceX,nsourceY);
 Eigen::MatrixXcd abxx(nsourceX,nsourceY);
 for (int ls = -lsmaxX; ls<=lsmaxX; ls++)
 {
 for (int ms = -lsmaxY; ms<=lsmaxY; ms++)
 {
 if((pow(ls*MX/dx,2)+pow(ms*MY/dy,2))<=pow(NA/lambda,2))
  {
    for (int i = 0; i < Nrangep; i++)
   {
    if ((lindexp[i] == ls) && (mindexp[i] == ms))
    {
     vcxx(ls+lsmaxX,ms+lsmaxY)=Axvc[ls+lsmaxX][ms+lsmaxY](i);
     abxx(ls+lsmaxX,ms+lsmaxY)=Axab[ls+lsmaxX][ms+lsmaxY](i);
    } 
   }
  }
 }
 }

 Eigen::MatrixXcd  phasexx(nsourceX,nsourceY);
 for(int is=0;is<nsourceX;is++)
 {
  for(int js=0;js<nsourceY;js++)
  {
   phasexx(is,js)=vcxx(is,js)/abs(vcxx(is,js));
  }
 }

 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 start=now;
 cout<<elapsed<<" Abs & ML reflectivity"<<endl;

  double mesh = 0.1; //degrees
  vector<double> dsx, dsy;
  int SDIV;
  double kmesh=k*mesh*(pi/180.);
  double skangx=k*NA/MX*sigma1;
  double skangy=k*NA/MY*sigma1;
  int l0max=skangx/kmesh+1;
  int m0max=skangy/kmesh+1;
  SDIV=0;
  for(int l=-l0max;l<=l0max;l++)
  {
    for(int m=-m0max;m<=m0max;m++)
    {
        double skx=l*kmesh;
         double sky=m*kmesh;
         double skxo=skx*MX;
         double skyo=sky*MY;
          if(((type==0)&&(skxo*skxo+skyo*skyo)<=pow(k*NA*sigma1,2))
             ||((type==1)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2))
            ||((type==2)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2)
               &&(abs(skyo)<=abs(skxo)*tan(pi*openangle/180./2.))))
        {
          dsx.push_back(skx);
          dsy.push_back(sky);
          SDIV++;
        }
     }
   }
  Eigen::MatrixXcd TCCXS0(ncut,ncut),TCCXSX(ncut,ncut),TCCXSY(ncut,ncut),
        TCCYS0(ncut,ncut),TCCYSX(ncut,ncut),TCCYSY(ncut,ncut);
  double sx,sy,kx,ky,ksx,ksy,kxp,kyp,ksxp,ksyp;
  complex<double> phase, phasep,sumxs0,sumxsx,sumxsy,sumys0,sumysx,sumysy;
  double pmax=pow(k*NA,2);
  for(int i=0; i<ncut; i++)
  {
    kx=2*pi/dx*linput[i];
    ky=2*pi/dy*minput[i];
     for(int j=0; j<ncut; j++)
     {
      kxp=2*pi/dx*linput[j];
      kyp=2*pi/dy*minput[j];
      sumxs0=0.;
      sumys0=0.;
      sumxsx=0.;
      sumysx=0.;
      sumxsy=0.;
      sumysy=0.;
 //       #pragma omp parallel for
       for (int is=0;is<SDIV;is++)
       {
        sx=dsx[is];
        sy=dsy[is];
        ksx=kx+sx;
        ksy=ky+sy;
        ksxp=kxp+sx;
        ksyp=kyp+sy;
        if(((MX*MX*ksx*ksx+MY*MY*ksy*ksy)<=pmax)&&((MX*MX*ksxp*ksxp+MY*MY*ksyp*ksyp)<=pmax))
        {
         phase=exp(zi*((ksx+sx0)*(ksx+sx0)+(ksy+sy0)*(ksy+sy0))/2./k*z0
           +zi*(MX*MX*ksx*ksx+MY*MY*ksy*ksy)/2./k*z);
         phasep=exp(zi*((ksxp+sx0)*(ksxp+sx0)+(ksyp+sy0)*(ksyp+sy0))/2./k*z0
           +zi*(MX*MX*ksxp*ksxp+MY*MY*ksyp*ksyp)/2./k*z);
          sumxs0=sumxs0+phase*conj(phasep)/(k*k-(sx0+sx)*(sx0+sx));
          sumys0=sumys0+phase*conj(phasep)/(k*k-(sy0+sy)*(sy0+sy));
          sumxsx=sumxsx+sx*phase*conj(phasep)/(k*k-(sx0+sx)*(sx0+sx));
          sumysx=sumysx+sx*phase*conj(phasep)/(k*k-(sy0+sy)*(sy0+sy));
          sumxsy=sumxsy+sy*phase*conj(phasep)/(k*k-(sx0+sx)*(sx0+sx));
          sumysy=sumysy+sy*phase*conj(phasep)/(k*k-(sy0+sy)*(sy0+sy));
         }
       }
       TCCXS0(i,j) =sumxs0/double(SDIV);
       TCCXSX(i,j) =sumxsx/double(SDIV);
       TCCXSY(i,j) =sumxsy/double(SDIV);
       TCCYS0(i,j) =sumys0/double(SDIV);
       TCCYSX(i,j) =sumysx/double(SDIV);
       TCCYSY(i,j) =sumysy/double(SDIV);
      }
   }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;

  vector <double> alphaXs0(ncut);
  vector<Eigen::VectorXcd> phipXs0(ncut,Eigen::VectorXcd (ncut));
  es.compute(TCCXS0);
  #pragma omp parallel for
  for(int i=0;i<ncut;i++)
  {
   alphaXs0[i]=es.eigenvalues()(i);
   for(int j=0;j<ncut;j++)
   {
    phipXs0[j](i)=es.eigenvectors()(i,j);
   }
  }
  vector <double> alphaXsx(ncut);
  vector<Eigen::VectorXcd> phipXsx(ncut,Eigen::VectorXcd (ncut));
  es.compute(TCCXSX);
  #pragma omp parallel for
  for(int i=0;i<ncut;i++)
  {
   alphaXsx[i]=es.eigenvalues()(i);
   for(int j=0;j<ncut;j++)
   {
    phipXsx[j](i)=es.eigenvectors()(i,j);
   }
  }

  vector <double> alphaXsy(ncut);
  vector<Eigen::VectorXcd> phipXsy(ncut,Eigen::VectorXcd (ncut));
  es.compute(TCCXSY);
  #pragma omp parallel for
  for(int i=0;i<ncut;i++)
  {
   alphaXsy[i]=es.eigenvalues()(i);
   for(int j=0;j<ncut;j++)
   {
    phipXsy[j](i)=es.eigenvectors()(i,j);
   }
  }

  vector <double> alphaYs0(ncut);
  vector<Eigen::VectorXcd> phipYs0(ncut,Eigen::VectorXcd (ncut));
  es.compute(TCCYS0);
  #pragma omp parallel for
  for(int i=0;i<ncut;i++)
  {
   alphaYs0[i]=es.eigenvalues()(i);
   for(int j=0;j<ncut;j++)
   {
    phipYs0[j](i)=es.eigenvectors()(i,j);
   }
  }

  vector <double> alphaYsx(ncut);
  vector<Eigen::VectorXcd> phipYsx(ncut,Eigen::VectorXcd (ncut));
  es.compute(TCCYSX);
  #pragma omp parallel for
  for(int i=0;i<ncut;i++)
  {
   alphaYsx[i]=es.eigenvalues()(i);
   for(int j=0;j<ncut;j++)
   {
    phipYsx[j](i)=es.eigenvectors()(i,j);
   }
  }

  vector <double> alphaYsy(ncut);
  vector<Eigen::VectorXcd> phipYsy(ncut,Eigen::VectorXcd (ncut));
  es.compute(TCCYSY);
  #pragma omp parallel for
  for(int i=0;i<ncut;i++)
  {
   alphaYsy[i]=es.eigenvalues()(i);
   for(int j=0;j<ncut;j++)
   {
    phipYsy[j](i)=es.eigenvectors()(i,j);
   }
  }

  vector<Eigen::VectorXcd> phiXs0(ncut,Eigen::VectorXcd (ncut));
  vector<Eigen::VectorXcd> phiXsx(ncut,Eigen::VectorXcd (ncut));
  vector<Eigen::VectorXcd> phiXsy(ncut,Eigen::VectorXcd (ncut));
  vector<Eigen::VectorXcd> phiYs0(ncut,Eigen::VectorXcd (ncut));
  vector<Eigen::VectorXcd> phiYsx(ncut,Eigen::VectorXcd (ncut));
  vector<Eigen::VectorXcd> phiYsy(ncut,Eigen::VectorXcd (ncut));

  vector <int> xs0(ncut),xsx(ncut),xsy(ncut),ys0(ncut),ysx(ncut),ysy(ncut);
  for(int i=0;i<ncut;i++)
  { 
    xs0[i]=i;
    xsx[i]=i;
    xsy[i]=i;
    ys0[i]=i;
    ysx[i]=i;
    ysy[i]=i;
  }

   for(int i=0;i<ncut-1;i++)
   {
     for(int j=ncut-1;i<j;j--)
     {
      if(abs(alphaXs0[j])>abs(alphaXs0[j-1]))
      {
       double alp=alphaXs0[j];
       alphaXs0[j]=alphaXs0[j-1];
       alphaXs0[j-1]=alp;
       int tmp=xs0[j];
       xs0[j]=xs0[j-1];
       xs0[j-1]=tmp;
      }
      if(abs(alphaXsx[j])>abs(alphaXsx[j-1]))
      {
       double alp=alphaXsx[j];
       alphaXsx[j]=alphaXsx[j-1];
       alphaXsx[j-1]=alp;
       int tmp=xsx[j];
       xsx[j]=xsx[j-1];
       xsx[j-1]=tmp;
      }
      if(abs(alphaXsy[j])>abs(alphaXsy[j-1]))
      {
       double alp=alphaXsy[j];
       alphaXsy[j]=alphaXsy[j-1];
       alphaXsy[j-1]=alp;
       int tmp=xsy[j];
       xsy[j]=xsy[j-1];
       xsy[j-1]=tmp;
      }
      if(abs(alphaYs0[j])>abs(alphaYs0[j-1]))
      {
       double alp=alphaYs0[j];
       alphaYs0[j]=alphaYs0[j-1];
       alphaYs0[j-1]=alp;
       int tmp=ys0[j];
       ys0[j]=ys0[j-1];
       ys0[j-1]=tmp;
      }
      if(abs(alphaYsx[j])>abs(alphaYsx[j-1]))
      {
       double alp=alphaYsx[j];
       alphaYsx[j]=alphaYsx[j-1];
       alphaYsx[j-1]=alp;
       int tmp=ysx[j];
       ysx[j]=ysx[j-1];
       ysx[j-1]=tmp;
      }
      if(abs(alphaYsy[j])>abs(alphaYsy[j-1]))
      {
       double alp=alphaYsy[j];
       alphaYsy[j]=alphaYsy[j-1];
       alphaYsy[j-1]=alp;
       int tmp=ysy[j];
       ysy[j]=ysy[j-1];
       ysy[j-1]=tmp;
      }
    }
   }
  for(int i=0;i<ncut;i++)
  {
   for(int j=0;j<ncut;j++)
   {
     phiXs0[i](j)=phipXs0[xs0[i]](j);
     phiXsx[i](j)=phipXsx[xsx[i]](j);
     phiXsy[i](j)=phipXsy[xsy[i]](j);
     phiYs0[i](j)=phipYs0[ys0[i]](j);
     phiYsx[i](j)=phipYsx[ysx[i]](j);
     phiYsy[i](j)=phipYsy[ysy[i]](j);
    }
  }

 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 start=now;
 cout<<elapsed<<" TCC + eigenvalue calculation"<<endl;

 for(int i=0;i<NDIVX;i++)
 for(int j=0;j<NDIVY;j++)
   imask>>mask2d[NDIVY*i+j]>>com;
 imask.getline(eol, sizeof(eol));

 double d0re[ncut],d0im[ncut],dxre[ncut],dxim[ncut],dyre[ncut],dyim[ncut];
 for(int n=0;n<ncut;n++)
  { 
    ipredict>>d0re[n];
    ipredict>>com;
   }
 for(int n=0;n<ncut;n++)
  {
    ipredict>>d0im[n];
    ipredict>>com;
  }
 for(int n=0;n<ncut;n++)
 {
  if(xinput[n]==1)
  {
    ipredict>>dxre[n];
    ipredict>>com;
  }
 }
 for(int n=0;n<ncut;n++)
 {
   if(xinput[n]==1)
   {
    ipredict>>dxim[n];
    ipredict>>com;
   }
  }
 for(int n=0;n<ncut;n++)
 {
  if(xinput[n]==1)
  {
   ipredict>>dyre[n];
   ipredict>>com;
  }
 }
 int ns=0;
 for(int n=0;n<ncut;n++)
 {
  if(xinput[n]==1)
  {
   ipredict>>dyim[n];
   ns=ns+1;
   if (ns<(ncuts-1)) ipredict>>com;
   }
 }
 ipredict.getline(eol, sizeof(eol));

 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 start=now;
 cout<<elapsed<<" Input mask pattern"<<endl;

 complex <double> ampab,ampvc;
 ampab=abxx(lsmaxX,lsmaxY);
 ampvc=vcxx(lsmaxX,lsmaxY);
 static double _Complex fpattern[FDIV][FDIV];
 #pragma omp parallel for 
 for (int i = 0; i < FDIV; i++)
 for (int j = 0; j < FDIV; j++)
 {
  pattern(i,j)=double(mask2d[FDIV*i+j])*(ampab-ampvc)+ampvc;
  fpattern[i][j] = real(pattern(i, j))+imag(pattern(i, j))*_Complex_I ;
 }  
 
 DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
 MKL_LONG status0;
 MKL_LONG dim_size0[2] = {FDIV, FDIV};
 status0 = DftiCreateDescriptor(&my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_size0);
 status0 = DftiCommitDescriptor(my_desc_handle);
 status0 = DftiComputeForward(my_desc_handle, fpattern);
 Eigen::MatrixXcd fmask(noutX,noutY);
 for (int i = 0; i < noutX; i++)
 {
  int l = (i - lpmaxX+FDIV)%FDIV;
  for (int j = 0; j < noutY; j++)
  {
   int m = (j - lpmaxY+FDIV)%FDIV;
   fmask(i, j) = (__real__ fpattern[l][m]+__imag__ fpattern[l][m]*zi)/double(FDIV)/double(FDIV);
  }
 }
 status0 = DftiFreeDescriptor(&my_desc_handle);

 Eigen::MatrixXcd  fampxx(noutX,noutY);
 {
   double kxs,kys,kxp,kyp;
   complex<double> phasesp;
   kxs=sx0;
   kys=sy0;
   for(int ip=0;ip<noutX;ip++)
   {
    kxp=2.*pi*(ip-lpmaxX)/dx;
    for(int jp=0;jp<noutY;jp++)
    {
     kyp=2.*pi*(jp-lpmaxY)/dy;
     phasesp=exp(-zi*(kxs*kxp+kxp*kxp/2.+kys*kyp+kyp*kyp/2.)/k*z0);
     fampxx(ip,jp)=fmask(ip,jp)*phasesp;
    }
   }
 }
 for(int ip=0;ip<noutX;ip++)
 {
  for(int jp=0;jp<noutY;jp++)
  {
   fampxx(ip,jp)=fampxx(ip,jp)/phasexx(lsmaxX,lsmaxY);
  }
 }

 Eigen::MatrixXcd a0xx(noutX,noutY),axxx(noutX,noutY),ayxx(noutX,noutY);
 for(int n=0;n<ncut;n++)
  {
    int ip = linput[n] + lpmaxX;
    int jp = minput[n] + lpmaxY;
    a0xx(ip, jp)=d0re[n]+zi*d0im[n];
   }
 for(int n=0;n<ncut;n++)
 {
  int ip = linput[n] + lpmaxX;
  int jp = minput[n] + lpmaxY;
     if(xinput[n]==1)
   {
    axxx(ip, jp)=(dxre[n]+zi*dxim[n]);
   }
   else
   {
    axxx(ip, jp)=0.;
    }
  }
 for(int n=0;n<ncut;n++)
 {
    int ip = linput[n] + lpmaxX;
    int jp = minput[n] + lpmaxY;
  if(xinput[n]==1)
  {
   ayxx(ip, jp)=(dyre[n]+zi*dyim[n]);
  }
  else
  {
   ayxx(ip, jp)=0.;
  }
 }
 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 start=now;
 cout<<elapsed<<" FT of mask pattern"<<endl;

 static double _Complex intensity[XDIV][XDIV];
 static double _Complex intensity0[XDIV][XDIV];
 vector< complex<double>>  Exs0(ncut),Eys0(ncut),Ezs0(ncut),Exsx(ncut),Eysx(ncut),Ezsx(ncut),Exsy(ncut),Eysy(ncut),Ezsy(ncut);
 vector< complex<double>>  Exsxy(ncut),Eysxy(ncut),Ezsxy(ncut);
 for (int ipl = 0; ipl<=0; ipl++)
// for (int ipl = 0; ipl<=1; ipl++)
 {
  for (int i = 0; i < ncut; i++)
  {
   double kxplus, kyplus, kzplus;
   kxplus = sx0 + 2 * pi*linput[i] / dx/2.;
   kyplus = sy0 + 2 * pi*minput[i] / dy/2.;
   kzplus = -sqrt(k*k - kxplus*kxplus-kyplus*kyplus);
   complex<double>  Ax,dxAx,dyAx,Ay,dxAy,dyAy;
   int ip,jp;
   ip=linput[i]+lpmaxX;
   jp=minput[i]+lpmaxY;
   double lp = ip - lpmaxX;
   double mp = jp-lpmaxY;

   if(ipl==0)
   {
    if(type==1)
    {
     Ax=fampxx(ip,jp);
     dxAx=0.;
     dyAx=0.;
    }
    else if(type==2)
    {
     Ax= fampxx(ip,jp)+a0xx(ip,jp);
     dxAx= axxx(ip,jp)*dx/2./pi;
     dyAx= ayxx(ip,jp)*dy/2./pi;
    }

     Exs0[i]=zi*k*Ax-zi/k*kxplus*kxplus*Ax;
     Exsx[i]=-2.*zi/k*kxplus*Ax+zi*k*dxAx-zi/k*kxplus*kxplus*dxAx;
     Exsy[i]=zi*k*dyAx-zi/k*kxplus*kxplus*dyAx;
     Exsxy[i]=Exsx[i]/(dx/2./pi)*lp/2.+Exsy[i]/(dy/2./pi)*mp/2.;
     Eys0[i]=-zi/k*kxplus*kyplus*Ax;
     Eysx[i]=-zi/k*kyplus*Ax-zi/k*kxplus*kyplus*dxAx;
     Eysy[i]=-zi/k*kxplus*Ax-zi/k*kxplus*kyplus*dyAx;
     Eysxy[i]=Eysx[i]/(dx/2./pi)*lp/2.+Eysy[i]/(dy/2./pi)*mp/2.;
     Ezs0[i]=-zi/k*kxplus*kzplus*Ax;
     Ezsx[i]=-zi/k*kzplus*Ax-zi/k*kxplus*kzplus*dxAx;
     Ezsy[i]=-zi/k*kxplus*kzplus*dyAx;
     Ezsxy[i]=Ezsx[i]/(dx/2./pi)*lp/2.+Ezsy[i]/(dy/2./pi)*mp/2.;
    }
   else if(ipl==1)
   {
    if(type==1)
    {
     Ay=fampxx(ip,jp);
     dxAy=0.;
     dyAy=0.;
    }
    else if(type==2)
    {
     Ay= fampxx(ip,jp)+a0xx(ip,jp);
     dxAy= axxx(ip,jp)*dx/2./pi;
     dyAy= ayxx(ip,jp)*dy/2./pi;
    }
     Exs0[i]=-zi/k*kxplus*kyplus*Ay;
     Exsx[i]=-zi/k*kyplus*Ay-zi/k*kxplus*kyplus*dxAy;
     Exsy[i]=-zi/k*kxplus*Ay-zi/k*kxplus*kyplus*dyAy;
     Exsxy[i]=Exsx[i]/(dx/2./pi)*lp/2.+Exsy[i]/(dy/2./pi)*mp/2.;
     Eys0[i]=zi*k*Ay-zi/k*kyplus*kyplus*Ay;
     Eysx[i]=zi*k*dxAy-zi/k*kyplus*kyplus*dxAy;
     Eysy[i]=-2.*zi/k*kyplus*Ay+zi*k*dyAy-zi/k*kyplus*kyplus*dyAy;
     Eysxy[i]=Eysx[i]/(dx/2./pi)*lp/2.+Eysy[i]/(dy/2./pi)*mp/2.;
     Ezs0[i]=-zi/k*kyplus*kzplus*Ay;
     Ezsx[i]=-zi/k*kyplus*kzplus*dxAy;
     Ezsy[i]=-zi/k*kzplus*Ay-zi/k*kyplus*kzplus*dyAy;
     Ezsxy[i]=Ezsx[i]/(dx/2./pi)*lp/2.+Ezsy[i]/(dy/2./pi)*mp/2.;
    }
  }

 static double _Complex intsmall[nout][nout];
//#pragma omp parallel for
 for(int i=0;i<nout;i++)
 for(int j=0;j<nout;j++)
    intsmall[i][j]=0.;

 DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
 MKL_LONG status;
 MKL_LONG dim_sizes[2] = {nout, nout};
 status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_sizes);
 status = DftiCommitDescriptor(my_desc1_handle);
 static double _Complex fnxs0[nout][nout],fnxs0x[nout][nout],fnxs0y[nout][nout],fnxsx[nout][nout],fnxsy[nout][nout],fnxsxy[nout][nout],
  fnys0[nout][nout],fnys0x[nout][nout],fnys0y[nout][nout], fnysx[nout][nout],fnysy[nout][nout],fnysxy[nout][nout],
  fnzs0[nout][nout],fnzs0x[nout][nout],fnzs0y[nout][nout],fnzsx[nout][nout],fnzsy[nout][nout],fnzsxy[nout][nout];
  complex <double> fxs0,fxs0x,fxs0y,fxsx,fxsy,fxsxy,fys0,fys0x,fys0y,fysx,fysy,fysxy,fzs0,fzs0x,fzs0y,fzsx,fzsy,fzsxy;
  double alphas0,alphasx,alphasy;
 for (int m=0;m<nsocs;m++)
 {
// #pragma omp parallel for
 for(int i=0;i<nout;i++)
 for(int j=0;j<nout;j++)
 {
  fnxs0[i][j]=0.;
  fnys0[i][j]=0.;
  fnzs0[i][j]=0.;
 }

 for (int n=0;n<ncut;n++)
 {
  if(ipl==0)
  {
    fxs0=Exs0[n]*phiXs0[m](n);
    fys0=Eys0[n]*phiXs0[m](n);
    fzs0=Ezs0[n]*phiXs0[m](n);
  }
  else if(ipl==1)
  {
    fxs0=Exs0[n]*phiYs0[m](n);
    fys0=Eys0[n]*phiYs0[m](n);
    fzs0=Ezs0[n]*phiYs0[m](n);
  }
   int ix=linput[n];
   int iy=minput[n];
   int px=(ix+nout)%nout;
   int py=(iy+nout)%nout;
   fnxs0[px][py]=real(fxs0)+imag(fxs0)*_Complex_I;
   fnys0[px][py]=real(fys0)+imag(fys0)*_Complex_I;
   fnzs0[px][py]=real(fzs0)+imag(fzs0)*_Complex_I;
 }
  status = DftiComputeBackward(my_desc1_handle, fnxs0);
  status = DftiComputeBackward(my_desc1_handle, fnys0);
  status = DftiComputeBackward(my_desc1_handle, fnzs0);
  if(ipl==0)
  {
   alphas0=alphaXs0[m];
  }
  else if(ipl==1)
  {
   alphas0=alphaYs0[m];
  }

// #pragma omp parallel for
  for(int i=0;i<nout;i++)
  for(int j=0;j<nout;j++)
  {
   intsmall[i][j] +=alphas0*(__real__ fnxs0[i][j]*__real__ fnxs0[i][j]
                            +__imag__ fnxs0[i][j]*__imag__ fnxs0[i][j]
                            +__real__ fnys0[i][j]*__real__ fnys0[i][j]
                            +__imag__ fnys0[i][j]*__imag__ fnys0[i][j]
                            +__real__ fnzs0[i][j]*__real__ fnzs0[i][j]
                            +__imag__ fnzs0[i][j]*__imag__ fnzs0[i][j]);
  }

 if (m<=nsocsxy)
 {
// #pragma omp parallel for
 for(int i=0;i<nout;i++)
 for(int j=0;j<nout;j++)
 {
  fnxs0x[i][j]=0.;
  fnxs0y[i][j]=0.;
  fnxsx[i][j]=0.;
  fnxsy[i][j]=0.;
  fnxsxy[i][j]=0.;
  fnys0x[i][j]=0.;
  fnys0y[i][j]=0.;
  fnysx[i][j]=0.;
  fnysy[i][j]=0.;
  fnysxy[i][j]=0.;
  fnzs0x[i][j]=0.;
  fnzs0y[i][j]=0.;
  fnzsx[i][j]=0.;
  fnzsy[i][j]=0.;
  fnzsxy[i][j]=0.;
 }
 for (int n=0;n<ncut;n++)
 {
  if(ipl==0)
  {
    fxs0x=Exs0[n]*phiXsx[m](n);
    fxs0y=Exs0[n]*phiXsy[m](n);
    fxsx=Exsx[n]*phiXsx[m](n);
    fxsy=Exsy[n]*phiXsy[m](n);
    fxsxy=Exsxy[n]*phiXs0[m](n);
    fys0x=Eys0[n]*phiXsx[m](n);
    fys0y=Eys0[n]*phiXsy[m](n);
    fysx=Eysx[n]*phiXsx[m](n);
    fysy=Eysy[n]*phiXsy[m](n);
    fysxy=Eysxy[n]*phiXs0[m](n);
    fzs0x=Ezs0[n]*phiXsx[m](n);
    fzs0y=Ezs0[n]*phiXsy[m](n);
    fzsx=Ezsx[n]*phiXsx[m](n);
    fzsy=Ezsy[n]*phiXsy[m](n);
    fzsxy=Ezsxy[n]*phiXs0[m](n);
  }
  else if(ipl==1)
  {
    fxs0x=Exs0[n]*phiYsx[m](n);
    fxs0y=Exs0[n]*phiYsy[m](n);
    fxsx=Exsx[n]*phiYsx[m](n);
    fxsy=Exsy[n]*phiYsy[m](n);
    fxsxy=Exsxy[n]*phiYs0[m](n);
    fys0x=Eys0[n]*phiYsx[m](n);
    fys0y=Eys0[n]*phiYsy[m](n);
    fysx=Eysx[n]*phiYsx[m](n);
    fysy=Eysy[n]*phiYsy[m](n);
    fysxy=Eysxy[n]*phiYs0[m](n);
    fzs0x=Ezs0[n]*phiYsx[m](n);
    fzs0y=Ezs0[n]*phiYsy[m](n);
    fzsx=Ezsx[n]*phiYsx[m](n);
    fzsy=Ezsy[n]*phiYsy[m](n);
    fzsxy=Ezsxy[n]*phiYs0[m](n);
  }
   int ix=linput[n];
   int iy=minput[n];
   int px=(ix+nout)%nout;
   int py=(iy+nout)%nout;
   fnxs0x[px][py]=real(fxs0x)+imag(fxs0x)*_Complex_I;
   fnxs0y[px][py]=real(fxs0y)+imag(fxs0y)*_Complex_I;
   fnxsx[px][py]=real(fxsx)+imag(fxsx)*_Complex_I;
   fnxsy[px][py]=real(fxsy)+imag(fxsy)*_Complex_I;
   fnxsxy[px][py]=real(fxsxy)+imag(fxsxy)*_Complex_I;
   fnys0x[px][py]=real(fys0x)+imag(fys0x)*_Complex_I;
   fnys0y[px][py]=real(fys0y)+imag(fys0y)*_Complex_I;
   fnysx[px][py]=real(fysx)+imag(fysx)*_Complex_I;
   fnysy[px][py]=real(fysy)+imag(fysy)*_Complex_I;
   fnysxy[px][py]=real(fysxy)+imag(fysxy)*_Complex_I;
   fnzs0x[px][py]=real(fzs0x)+imag(fzs0x)*_Complex_I;
   fnzs0y[px][py]=real(fzs0y)+imag(fzs0y)*_Complex_I;
   fnzsx[px][py]=real(fzsx)+imag(fzsx)*_Complex_I;
   fnzsy[px][py]=real(fzsy)+imag(fzsy)*_Complex_I;
   fnzsxy[px][py]=real(fzsxy)+imag(fzsxy)*_Complex_I;
 }
  status = DftiComputeBackward(my_desc1_handle, fnxs0x);
  status = DftiComputeBackward(my_desc1_handle, fnxs0y);
  status = DftiComputeBackward(my_desc1_handle, fnxsx);
  status = DftiComputeBackward(my_desc1_handle, fnxsy);
  status = DftiComputeBackward(my_desc1_handle, fnxsxy);
  status = DftiComputeBackward(my_desc1_handle, fnys0x);
  status = DftiComputeBackward(my_desc1_handle, fnys0y);
  status = DftiComputeBackward(my_desc1_handle, fnysx);
  status = DftiComputeBackward(my_desc1_handle, fnysy);
  status = DftiComputeBackward(my_desc1_handle, fnysxy);
  status = DftiComputeBackward(my_desc1_handle, fnzs0x);
  status = DftiComputeBackward(my_desc1_handle, fnzs0y);
  status = DftiComputeBackward(my_desc1_handle, fnzsx);
  status = DftiComputeBackward(my_desc1_handle, fnzsy);
  status = DftiComputeBackward(my_desc1_handle, fnzsxy);

  if(ipl==0)
  {
   alphasx=alphaXsx[m];
   alphasy=alphaXsy[m];
  }
  else if(ipl==1)
  {
   alphasx=alphaYsx[m];
   alphasy=alphaYsy[m];
  }

// #pragma omp parallel for
  for(int i=0;i<nout;i++)
  for(int j=0;j<nout;j++)
  {
   intsmall[i][j] +=2.*alphasx*(__real__ fnxs0x[i][j]*__real__ fnxsx[i][j]
                            +__imag__ fnxs0x[i][j]*__imag__ fnxsx[i][j]
                            +__real__ fnys0x[i][j]*__real__ fnysx[i][j]
                            +__imag__ fnys0x[i][j]*__imag__ fnysx[i][j]
                            +__real__ fnzs0x[i][j]*__real__ fnzsx[i][j]
                            +__imag__ fnzs0x[i][j]*__imag__ fnzsx[i][j])
                +2.*alphasy*(__real__ fnxs0y[i][j]*__real__ fnxsy[i][j]
                            +__imag__ fnxs0y[i][j]*__imag__ fnxsy[i][j]
                            +__real__ fnys0y[i][j]*__real__ fnysy[i][j]
                            +__imag__ fnys0y[i][j]*__imag__ fnysy[i][j]
                            +__real__ fnzs0y[i][j]*__real__ fnzsy[i][j]
                            +__imag__ fnzs0y[i][j]*__imag__ fnzsy[i][j])
                +2.*alphas0*(__real__ fnxs0[i][j]*__real__ fnxsxy[i][j]
                            +__imag__ fnxs0[i][j]*__imag__ fnxsxy[i][j]
                            +__real__ fnys0[i][j]*__real__ fnysxy[i][j]
                            +__imag__ fnys0[i][j]*__imag__ fnysxy[i][j]
                            +__real__ fnzs0[i][j]*__real__ fnzsxy[i][j]
                            +__imag__ fnzs0[i][j]*__imag__ fnzsxy[i][j]);
  }
 }
 }
 status = DftiComputeForward(my_desc1_handle, intsmall);
 status = DftiFreeDescriptor(&my_desc1_handle);

#pragma omp parallel for
 for(int i=0;i<XDIV;i++)
 for(int j=0;j<XDIV;j++)
    intensity[i][j]=0.;

 for(int i=0;i<nout/2;i++)
  {
    for(int j=0;j<nout/2;j++)
     intensity[i][j]=intsmall[i][j]/nout/nout;
    for(int j=nout/2;j<nout;j++)
     intensity[i][XDIV-nout+j]=intsmall[i][j]/nout/nout;
  }
 for(int i=nout/2;i<nout;i++)
  {
    for(int j=0;j<nout/2;j++)
     intensity[XDIV-nout+i][j]=intsmall[i][j]/nout/nout;
    for(int j=nout/2;j<nout;j++)
     intensity[XDIV-nout+i][XDIV-nout+j]=intsmall[i][j]/nout/nout;
  }

  if(ipl==0)
  {
#pragma omp parallel for
   for(int i=0;i<XDIV;i++)
   for(int j=0;j<XDIV;j++)
    intensity0[i][j]=intensity[i][j];
  }
 }

 DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
 MKL_LONG status;
 MKL_LONG dim_sizes2[2] = {XDIV, XDIV};
 status = DftiCreateDescriptor(&my_desc2_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_sizes2);
 status = DftiCommitDescriptor(my_desc2_handle);
 status = DftiComputeBackward(my_desc2_handle, intensity0);
 status = DftiComputeBackward(my_desc2_handle, intensity);
 status = DftiFreeDescriptor(&my_desc2_handle);

 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 start=now;
 cout<<elapsed<<" Image intensity integration by SOCS"<<endl;

 ofsint<<"data,1"<<endl;
 ofsint<<"memo1"<<endl;
 ofsint<<"memo2"<<endl;
 for(int i=0;i<XDIV;i++)
 {
  double x=i*dx/double(XDIV);
  ofsint<<","<<x/MX;
 }
 ofsint<<endl;

 for(int j=0;j<XDIV;j++)
 {
  double y = j*dy / double(XDIV);
  ofsint << y/MY;
  for(int i=0;i<XDIV;i++)
  {
   ofsint << ","<<__real__ intensity0[i][j];
//  ofsint << ","<<(__real__ intensity0[i][j]+__real__ intensity[i][j])/2.;
  }
  ofsint << endl;
 }
 ofsint.close();

 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 cout<<elapsed<<" Output image intensity"<<endl;

 return 0;
}
