#include <iostream>
#include <fstream>
#include <cstdlib>
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
#define XDIV 512  // XDIV=NDIVX/MX

void source(double NA,int type,double sigma1,double sigma2,double openangle,double k,double dx,double dy,
int&ndivs, vector<vector<vector<int>>>& l0s,vector<vector<vector<int>>>& m0s,vector<vector<int>>& SDIV,int& MX,int& MY);

int main (int argc,char* argv[])
{
 magma_init();
 double pi=atan(1.)*4.;
 complex<double> zi (0., 1.);
 char eol[512];
 char com;

 ifstream imask("mask.csv"); //input mask pattern
 ofstream ofsint("emint.csv"); //output image intensity

 int MX=4; //X magnification
 int MY=4; //Y magnification
 int NDIVX=2048; //X pitch (nm)
 int NDIVY=NDIVX; //Y pitch (nm)
 double dx =NDIVX;
 double dy = NDIVY;

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

 double sigmadiv=2.; //division angle of the source (degree)
 int ndivs=max(1,int(180./pi*lambda/dx/sigmadiv));

 vector<vector<vector<int>>> l0s(ndivs,vector<vector<int>>(ndivs,vector<int>()));
 vector<vector<vector<int>>> m0s(ndivs,vector<vector<int>>(ndivs,vector<int>()));
 vector<vector<int>> SDIV(ndivs,vector<int>(ndivs));
 source(NA,type,sigma1,sigma2,openangle,k,dx,dy,ndivs,l0s,m0s,SDIV,MX,MY);
 int SDIVMAX=0;
 int SDIVSUM=0;
 for(int nsx=0;nsx<ndivs;nsx++)
 for(int nsy=0;nsy<ndivs;nsy++)
 {
  SDIVMAX=max(SDIVMAX,SDIV[nsx][nsy]);
  SDIVSUM=SDIVSUM+SDIV[nsx][nsy];
 }

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
 int nsourceXL=2*lsmaxX+10;
 int nsourceYL=2*lsmaxY+10;
 int noutXL=2*lpmaxX+10;
 int noutYL=2*lpmaxY+10;

 int FDIVX1 = FDIVX + 1;
 int FDIVY1 = FDIVY + 1;
 vector< complex<double> > cexpX(FDIVX1),cexpY(FDIVY1);
 exponential(cexpX, pi, FDIVX);
 exponential(cexpY, pi, FDIVY);

 double cutx,cuty;
 cutx = NA/MX*6.;
 cuty = NA/MY*6.;;
 int LMAX = cutx*dx / lambda;
 int Lrange = 2 * LMAX + 1;
 int Lrange2 = 4 * LMAX + 1;
 int MMAX = cuty*dy / lambda;
 int Mrange = 2 * MMAX + 1;
 int Mrange2 = 4 * MMAX + 1;

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

 int ncut=0;
 vector<int> linput(Nrange),minput(Nrange);
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
    linput[ncut]=ip-lpmaxX;
    minput[ncut]=jp-lpmaxY;
    ncut++;
   }
  }
 }

 for(int i=0;i<NDIVX;i++)
 for(int j=0;j<NDIVY;j++)
 imask>>mask2d[NDIVY*i+j]>>com;
 imask.getline(eol, sizeof(eol));

 vector<vector<vector<vector<vector<double>>>>> isum(ndivs,vector<vector<vector<vector<double>>>>
  (ndivs,vector<vector<vector<double>>>(XDIV,vector<vector<double>>(XDIV,vector<double>(SDIVMAX)))));
 vector<vector<vector<vector<vector<double>>>>> isum0(ndivs,vector<vector<vector<vector<double>>>>
  (ndivs,vector<vector<vector<double>>>(XDIV,vector<vector<double>>(XDIV,vector<double>(SDIVMAX)))));

 DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
 MKL_LONG status;
 MKL_LONG dim_sizes[2] = {XDIV, XDIV};
 status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_sizes);
 status = DftiCommitDescriptor(my_desc1_handle);

for(int nsx=0;nsx<ndivs;nsx++)
for(int nsy=0;nsy<ndivs;nsy++)
{
 double kx0,ky0,sx0, sy0;
 kx0 = k*sin(pi / 180.*theta0)*cos(pi / 180.*phi0);
 ky0 = k*sin(pi / 180.*theta0)*sin(pi / 180.*phi0);
 sx0 = 2.*pi/dx*nsx/double(ndivs)+kx0;
 sy0 = 2.*pi/dy*nsy/double(ndivs)+ky0;

// ampS calculates the diffraction amplitude (vector potential). This routine in included in header.h.
// The input mask pattern is mask2d and the output amplitude is Ax.
// 'X' specifies the Ax polarization. 'Y' will calculates Ay polarization.

 vector<vector<Eigen::VectorXcd>> Ax(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrange)));
 ampS('X',Ax, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrange, lindex, mindex, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, sx0, sy0, eabs, dabs, cexpX,cexpY);

 vector<vector<Eigen::MatrixXcd>>  ampxx(nsourceXL, vector<Eigen::MatrixXcd>(nsourceYL));
 for(int is=0;is<nsourceXL;is++)
 for(int js=0;js<nsourceYL;js++)
  {
    ampxx[is][js].resize(noutXL,noutYL);
    for(int ip=0;ip<noutXL;ip++)
    for(int jp=0;jp<noutYL;jp++)
          ampxx[is][js](ip,jp)=-1000.;
 } 

  for(int is=0;is<nsourceXL;is++)
  {
   for(int js=0;js<nsourceYL;js++)
   {
    if((pow((is-lsmaxX)*MX/dx,2)+pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)*1.0)
    {
     for(int n=0;n<Nrange;n++)
     {
      int ip=lindex[n]-(is-lsmaxX)+lpmaxX;
      int jp=mindex[n]-(js-lsmaxY)+lpmaxY;
      if((0<=ip)&&(ip<noutXL)&&(0<=jp)&&(jp<noutYL))
       ampxx[is][js](ip,jp)=Ax[is][js](n);
     }
    }
   }
  }

 vector<Eigen::VectorXcd> Ex0m(SDIV[nsx][nsy], Eigen::VectorXcd(ncut)), Ey0m(SDIV[nsx][nsy], Eigen::VectorXcd(ncut)),
   Ez0m(SDIV[nsx][nsy], Eigen::VectorXcd(ncut));
 for (int ipl = 0; ipl<=0; ipl++)
// for (int ipl = 0; ipl<=1; ipl++)
 {
 for (int is = 0; is < SDIV[nsx][nsy]; is++)
 {
  double kx, ky;
  kx = sx0 + 2.*pi/dx*l0s[nsx][nsy][is];
  ky = sy0 + 2.*pi/dy*m0s[nsx][nsy][is];
  int ls=l0s[nsx][nsy][is]+lsmaxX;
  int ms=m0s[nsx][nsy][is]+lsmaxY;
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
   if(ipl==0)
   {
    if(real(ampxx[ls][ms](ip,jp))<-100) cout<<ip-lpmaxX+ls-lsmaxX<<","<<jp-lpmaxY+ms-lsmaxY<<endl;
    Ax= ampxx[ls][ms](ip,jp)/sqrt(k*k-kx*kx);
    Ay=0.;
   }
   else if(ipl==1)
   {
    Ay= ampxx[ls][ms](ip,jp)/sqrt(k*k-ky*ky);
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

 for (int is = 0; is < SDIV[nsx][nsy]; is++)
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
  kxn = 2.*pi/dx*nsx/double(ndivs)+2.*pi/dx*l0s[nsx][nsy][is]  + 2.*pi*linput[n] / dx ;
  kyn = 2.*pi/dy*nsy/double(ndivs)+2.*pi/dy*m0s[nsx][nsy][is]  + 2.*pi*minput[n] / dy ;
  if ((MX*MX*kxn*kxn+MY*MY*kyn*kyn) <= pow(NA*k,2))
  {
   complex<double> phase;
   phase=exp(zi*((kxn+kx0)*(kxn+kx0)+(kyn+ky0)*(kyn+ky0))/2./k*z0+zi*(MX*MX*kxn*kxn+MY*MY*kyn*kyn)/2./k*double(z));
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
   isum[nsx][nsy][i][j][is] = __real__ fnx[i][j]*__real__ fnx[i][j]
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
   for (int is = 0; is < SDIV[nsx][nsy]; is++)
   isum0[nsx][nsy][i][j][is] =isum[nsx][nsy][i][j][is];
 }
 }
 }
 status = DftiFreeDescriptor(&my_desc1_handle);

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
   double sum(0.);
   for(int nsx=0;nsx<ndivs;nsx++)
   for(int nsy=0;nsy<ndivs;nsy++)
   {
    for (int is = 0; is < SDIV[nsx][nsy]; is++)
    {
    sum += isum0[nsx][nsy][i][j][is];
//    sum += isum0[nsx][nsy][i][j][is]+isum[nsx][nsy][i][j][is];
    }
   }
    ofsint << ","<<sum/SDIVSUM;
//   ofsint << ","<<sum /2./SDIVSUM;
  }
  ofsint << endl;
 }
 ofsint.close();
 return 0;
}

void source(double NA,int type,double sigma1,double sigma2,double openangle,double k,double dx,double dy,
int& ndivs, vector<vector<vector<int>>>& l0s,vector<vector<vector<int>>>& m0s,vector<vector<int>>& SDIV,int& MX,int& MY)
{
 double pi=atan(1.)*4.;
 double dkxang=2.*pi/dx;
 double dkyang=2.*pi/dy;
 double skangx=k*NA/MX*sigma1;
 double skangy=k*NA/MY*sigma1;
 int l0max=skangx/dkxang+1;
 int m0max=skangy/dkyang+1;

 for(int nsx=0;nsx<ndivs;nsx++)
 for(int nsy=0;nsy<ndivs;nsy++)
 {
  SDIV[nsx][nsy]=0;
   for(int l=-l0max;l<=l0max;l++)
   for(int m=-m0max;m<=m0max;m++)
    {
      double skx=l*dkxang+2.*pi/dx*nsx/double(ndivs);
      double sky=m*dkyang+2.*pi/dy*nsy/double(ndivs);
      double skxo=skx*MX;
      double skyo=sky*MY;
      if(((type==0)&&(skxo*skxo+skyo*skyo)<=pow(k*NA*sigma1,2))
         ||((type==1)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2))
         ||((type==2)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2)
	      &&(abs(skyo)<=abs(skxo)*tan(pi*openangle/180./2.))))
        {
         l0s[nsx][nsy].push_back(l);
         m0s[nsx][nsy].push_back(m);
         SDIV[nsx][nsy]++;
        }
    }
  }
}


