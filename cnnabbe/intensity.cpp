#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#define MKL_NUM_THREADS 100
#include "mkl_dfti.h"
#define _USE_MATH_DEFINES
#include <omp.h>
using namespace std;
#define EIGEN_USE_MKL_ALL
#include "Eigen/Eigen"
#include "magma_v2.h"
#include "../include/header.h"

#define _Complex_I (1.0iF) 
#define XDIV 512  //XDIV=NDIVX/MX
#define FDIV 2048 //FDIV=NDIVX

int main (int argc,char* argv[])
{
 double pi=atan(1.)*4.;
 complex<double> zi (0., 1.);
 char eol[512];
 char com;

 ifstream imask("./mask.csv"); //input mask pattern
 ifstream ipredict("nnpredict.csv"); //input M3D parameters
 string ftint, intensity;
 ftint="ftint.csv"; //output image intensity by FT
 intensity="nnabbe.csv"; //output image intensity by CNN

 int ncut=1901; //number of the (l,m) pairs for a0
 int ncuts=1749; //number of the (l,m) pairs for ax and ay

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

 double mesh=0.1; //degrees
 vector<double> dkx, dky;
 int SDIV=0;
 double kmesh=k*mesh*(pi/180.);
 double skangx=k*NA/MX*sigma1;
 double skangy=k*NA/MY*sigma1;
 int l0max=skangx/kmesh+1;
 int m0max=skangy/kmesh+1;
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
      dkx.push_back(skx);
      dky.push_back(sky);
      SDIV++;
     }
   }
 }

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
 cuty = NA/MY*6.;;
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

vector<Eigen::VectorXcd> Ex0m(SDIV, Eigen::VectorXcd(ncut)), Ey0m(SDIV, Eigen::VectorXcd(ncut)), Ez0m(SDIV, Eigen::VectorXcd(ncut));
for(int type=1;type<=2;type++)
{
 string filename;
 if(type==1) filename=ftint;
 else if(type==2) filename=intensity;
 ofstream ofsint(filename);
 ofsint<<"data,1"<<endl;
 ofsint<<"memo1"<<endl;
 ofsint<<"memo2"<<endl;
 for(int i=0;i<XDIV;i++)
 {
  double x=i*dx/double(XDIV);
  ofsint<<","<<x/MX;
 }
 ofsint<<endl;

 vector<vector<vector<double>>> isum(XDIV,vector<vector<double>>(XDIV,vector<double>(SDIV)));
 vector<vector<vector<double>>> isum0(XDIV,vector<vector<double>>(XDIV,vector<double>(SDIV)));
 DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
 MKL_LONG status;
 MKL_LONG dim_sizes[2] = {XDIV, XDIV};
 status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_sizes);
 status = DftiCommitDescriptor(my_desc1_handle);

 for (int ipl = 0; ipl<=0; ipl++)
 {
 for (int is = 0; is < SDIV; is++)
 {
  double kx, ky, cosphi, sinphi;
  kx = sx0 + dkx[is];
  ky = sy0 + dky[is];
  double ls=dkx[is]/(2.*pi/dx);
  double ms=dky[is]/(2.*pi/dy);
  for (int i = 0; i < ncut; i++)
  {
   double kxplus, kyplus, kxy2;
   complex<double> klm;
   kxplus = kx + 2 * pi*linput[i] / dx;
   kyplus = ky + 2 * pi*minput[i] / dy;
   kxy2 = pow(kxplus, 2) + pow(kyplus, 2);
   klm = sqrt(k*k - kxy2);
   complex<double>  Ax, Ay;
   int ip,jp;
   ip=linput[i]+lpmaxX;
   jp=minput[i]+lpmaxY;
   double lp = ip - lpmaxX;
   double mp = jp-lpmaxY;
   if(ipl==0)
   {
    if(type==1)
    {
     Ax=fampxx(ip,jp)/sqrt(k*k-kx*kx);
    }
    else if(type==2)
    {
     Ax= (fampxx(ip,jp)+a0xx(ip,jp)+axxx(ip,jp)*(ls+lp/2.)+ayxx(ip,jp)*(ms+mp/2.))/sqrt(k*k-kx*kx);
    }
     Ay=0.;
   }
   else if(ipl==1)
   {
    if(type==1)
    {
     Ay=fampxx(ip,jp)/sqrt(k*k-ky*ky);
    }
    else if(type==2)
    {
     Ay= (fampxx(ip,jp)+a0xx(ip,jp)+axxx(ip,jp)*(ls+lp/2.)+ayxx(ip,jp)*(ms+mp/2.))/sqrt(k*k-ky*ky);
    }
    Ax=0.;
    }

   complex<double>  EAx, EAy, EAz;
   EAx=zi*k*Ax-zi/k*(pow(kxplus,2.)*Ax+kxplus*kyplus*Ay);
   EAy=zi*k*Ay-zi/k*(kxplus*kyplus*Ax+pow(kyplus,2.)*Ay);
   EAz=zi*klm/k*(kxplus*Ax+kyplus*Ay);

   Ex0m[is](i) = EAx;
   Ey0m[is](i) = EAy;
   Ez0m[is](i) = EAz;

/*  For high NA optics

     complex<double> Exm = EAx;
     complex<double> Eym = EAy;
     complex<double> Ezm = EAz;
     complex<double> ETEm, ETMm, Ex0mn,Ey0mn,Ez0mn;
     double eps=0.0000000001;
     if (abs(kxplus-sx0)<eps && abs(kyplus-sy0)<eps)
     {
	Ex0mn = sqrt(abs(Exm)*abs(Exm) + abs(Eym)*abs(Eym) + abs(Ezm)*abs(Ezm))
          /sqrt(abs(Exm)*abs(Exm) + abs(Eym)*abs(Eym)) * Exm;
	Ey0mn = sqrt(abs(Exm)*abs(Exm) + abs(Eym)*abs(Eym) + abs(Ezm)*abs(Ezm))
          /sqrt(abs(Exm)*abs(Exm) + abs(Eym)*abs(Eym)) * Eym;
	Ez0mn = 0.;
     }
     else
     {
	//----------------Mask-------------
	double kz0 = -sqrt(k*k - sx0*sx0 - sy0*sy0);

	//k1
	double kx1 = kxplus;
	double ky1 = kyplus;
	double kz1 = -sqrt(k*k - kx1*kx1 - ky1*ky1);

	//n ~ k0 x k1
	double AAx = sy0 * kz1 - kz0 * ky1;
	double AAy = kz0 * kx1 - sx0 * kz1;
	double AAz = sx0 * ky1 - sy0 * kx1;
        double norm=sqrt(AAx*AAx + AAy*AAy + AAz*AAz);
	double nAx = AAx /norm ;
	double nAy = AAy /norm;
	double nAz = AAz /norm;

	//t ~ k1 x n
	double Bx = ky1 * nAz - kz1 * nAy;
	double By = kz1 * nAx - kx1 * nAz;
	double Bz = kx1 * nAy - ky1 * nAx;
        norm=sqrt(Bx*Bx + By*By + Bz*Bz);
	double nBx = Bx /norm;
	double nBy = By /norm;
	double nBz = Bz /norm;

	//ETE, ETM
	ETEm = Exm * nAx + Eym * nAy + Ezm * nAz;
	ETMm = Exm * nBx + Eym * nBy + Ezm * nBz;

	//----------------Wafer-------------

	//k'
	double kx2 = MX * (kxplus-sx0);
	double ky2 = MY* (kyplus-sy0);
	double kz2 = -sqrt(k*k - kx2*kx2 - ky2*ky2);

	//n'~ ez x k'
	double Cx = ky2;
	double Cy = -kx2;
	double Cz = 0;
        norm=sqrt(Cx*Cx + Cy*Cy + Cz*Cz);
	double nCx = Cx /norm;
	double nCy = Cy /norm;
	double nCz = Cz /norm;

	//t'~k' x n'
	double Dx = ky2 * nCz - kz2 * nCy;
	double Dy = kz2 * nCx - kx2 * nCz;
	double Dz = kx2 * nCy - ky2 * nCx;
        norm=sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
	double nDx = Dx / norm;
	double nDy = Dy / norm;
	double nDz = Dz / norm;

	//Ex, Ey, Ez
	Ex0mn = ETEm * nCx + ETMm * nDx;
	Ey0mn = ETEm * nCy + ETMm * nDy;
	Ez0mn = ETEm * nCz + ETMm * nDz;
      }

   Ex0m[is](i) = Ex0mn;
   Ey0m[is](i) = Ey0mn;
   Ez0m[is](i) = Ez0mn;
*/
  }
 }

 for (int is = 0; is < SDIV; is++)
 {
 static double _Complex fnx[XDIV][XDIV],fny[XDIV][XDIV],fnz[XDIV][XDIV]; 
 #pragma omp parallel for
 for(int i=0;i<XDIV;i++)
 for(int j=0;j<XDIV;j++)
 {
  fnx[i][j]=0.;
  fny[i][j]=0.;
  fnz[i][j]=0.;
 }
 complex <double> fx,fy,fz;
 double kxn,kyn;
 for (int n = 0; n < ncut; n++)
 {
  kxn = dkx[is] + 2.*pi*linput[n] / dx ;
  kyn = dky[is] + 2.*pi*minput[n] / dy ;
  if ((MX*MX*kxn*kxn+MY*MY*kyn*kyn) <= pow(NA*k,2))
  {
   complex<double> phase;
   phase=exp(zi*((kxn+sx0)*(kxn+sx0)+(kyn+sy0)*(kyn+sy0))/2./k*z0+zi*(MX*MX*kxn*kxn+MY*MY*kyn*kyn)/2./k*double(z));
   fx=Ex0m[is](n)*phase;
   fy=Ey0m[is](n)*phase;
   fz=Ez0m[is](n)*phase;
   int ix=linput[n];
   int iy=minput[n];
   int px=(ix+XDIV)%XDIV;
   int py=(iy+XDIV)%XDIV;
   fnx[px][py]=real(fx)+imag(fx)*_Complex_I;
   fny[px][py]=real(fy)+imag(fy)*_Complex_I;
   fnz[px][py]=real(fz)+imag(fz)*_Complex_I;
  }
 }
 status = DftiComputeBackward(my_desc1_handle, fnx);
 status = DftiComputeBackward(my_desc1_handle, fny);
 status = DftiComputeBackward(my_desc1_handle, fnz);
 #pragma omp parallel for
 for(int i=0;i<XDIV;i++)
 for(int j=0;j<XDIV;j++)
 {
   isum[i][j][is] = __real__ fnx[i][j]*__real__ fnx[i][j]
                 +__imag__ fnx[i][j]*__imag__ fnx[i][j]
                 +__real__ fny[i][j]*__real__ fny[i][j]
                 +__imag__ fny[i][j]*__imag__ fny[i][j]
                 +__real__ fnz[i][j]*__real__ fnz[i][j]
                 +__imag__ fnz[i][j]*__imag__ fnz[i][j];
 }
 }
 if(ipl==0)
 {
   #pragma omp parallel for
   for(int i=0;i<XDIV;i++)
   for(int j=0;j<XDIV;j++)
   for (int is = 0; is < SDIV; is++)
   isum0[i][j][is] =isum[i][j][is];
 }
 }
 status = DftiFreeDescriptor(&my_desc1_handle);

 for(int j=0;j<XDIV;j++)
 {
  double y = j*dy / double(XDIV);
  ofsint << y/MY;
  for(int i=0;i<XDIV;i++)
  {
   double sum(0.);
//   #pragma omp parallel for reduction(+:sum)
   for (int is = 0; is < SDIV; is++)
   {
    sum += isum0[i][j][is];
//    sum += isum0[i][j][is]+isum[i][j][is];
   }
    ofsint << ","<<sum / SDIV;
//    ofsint << ","<<sum / SDIV/2.;
  }
   ofsint << endl;
 }
  ofsint.close();
}
 return 0;
}

